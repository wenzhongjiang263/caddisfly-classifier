"""Reusable inference helpers for the packaged caddisfly classifier."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from caddisfly_v2.model import build_model
from caddisfly_v2.transforms import build_transform


PACKAGE_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PACKAGE_ROOT / "model"
CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"
ACTIVE_MODEL_PATH = MODEL_DIR / "active_model.json"

DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESIZE_SIZE = 256
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]
SUPPORTED_MODEL_NAME = "mobilenet_v2"


class InferenceError(RuntimeError):
    """Raised when model loading or prediction fails with a clear message."""


@dataclass(frozen=True)
class ModelArtifacts:
    model: nn.Module
    preprocess: transforms.Compose
    class_names: List[str]
    config: Dict[str, Any]
    device: torch.device
    artifact_version: int = 1
    labels: Dict[str, Any] = field(default_factory=dict)
    detector_models: Tuple[Any, Any] | None = None


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise InferenceError(f"Required file is missing: {path}")
    if not path.is_file():
        raise InferenceError(f"Required path is not a file: {path}")
    try:
        with path.open("r", encoding="utf-8") as json_file:
            return json.load(json_file)
    except json.JSONDecodeError as exc:
        raise InferenceError(f"Could not parse JSON file: {path}") from exc


def _build_mobilenet_v2(num_classes: int) -> nn.Module:
    try:
        model = models.mobilenet_v2(weights=None)
    except TypeError:
        model = models.mobilenet_v2(pretrained=False)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def _build_preprocess(model_config: Dict[str, Any]) -> transforms.Compose:
    image_size = int(model_config.get("image_size", DEFAULT_IMAGE_SIZE))
    resize_size = int(model_config.get("resize_size", DEFAULT_RESIZE_SIZE))
    mean = model_config.get("normalization_mean", DEFAULT_IMAGENET_MEAN)
    std = model_config.get("normalization_std", DEFAULT_IMAGENET_STD)

    if not isinstance(mean, list) or not isinstance(std, list) or len(mean) != 3 or len(std) != 3:
        raise InferenceError("Model configuration must contain three-channel normalization mean and std.")

    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise InferenceError(f"Model checkpoint is missing: {path}")
    if not path.is_file():
        raise InferenceError(f"Model checkpoint path is not a file: {path}")
    try:
        # The packaged checkpoint is a trusted local full checkpoint, not a
        # weights-only file. PyTorch 2.6+ defaults can reject it otherwise.
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    except Exception as exc:
        raise InferenceError(f"Failed to load model checkpoint: {path}") from exc

    if not isinstance(checkpoint, dict):
        raise InferenceError("Model checkpoint must be a dictionary checkpoint.")
    return checkpoint


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise InferenceError("CUDA was requested, but this PyTorch environment cannot access it.")
    if device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise InferenceError("MPS was requested, but this PyTorch environment cannot access it.")
    return device


def _load_v2_artifacts(checkpoint_path: Path, device: torch.device) -> ModelArtifacts:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        raise InferenceError(f"Failed to load v2 model checkpoint: {checkpoint_path}") from exc
    if not isinstance(checkpoint, dict):
        raise InferenceError("The v2 checkpoint must be a dictionary.")
    metadata = checkpoint.get("metadata")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(metadata, dict) or metadata.get("artifact_version") != 2:
        raise InferenceError("The selected checkpoint is not a version-2 artifact.")
    if not isinstance(state_dict, dict):
        raise InferenceError("The v2 checkpoint does not contain model_state_dict.")
    labels = metadata.get("labels")
    if not isinstance(labels, dict) or not isinstance(labels.get("species"), list):
        raise InferenceError("The v2 checkpoint has no valid structured species labels.")
    class_names = labels["species"]
    manifest_path = checkpoint_path.with_name("model_manifest.json")
    deployment_manifest = _read_json(manifest_path)
    if not isinstance(deployment_manifest, dict):
        raise InferenceError("The v2 deployment manifest must contain an object.")
    if deployment_manifest.get("artifact_version") != 2:
        raise InferenceError("The v2 deployment manifest has an invalid artifact version.")
    checkpoint_descriptor = deployment_manifest.get("checkpoint")
    if not isinstance(checkpoint_descriptor, dict):
        raise InferenceError("The v2 deployment manifest has no checkpoint descriptor.")
    if checkpoint_descriptor.get("file") != checkpoint_path.name:
        raise InferenceError("The active checkpoint name does not match its deployment manifest.")
    if int(checkpoint_descriptor.get("bytes", -1)) != checkpoint_path.stat().st_size:
        raise InferenceError("The active checkpoint size does not match its deployment manifest.")
    for field_name in ("backbone", "image_size", "crop_mode"):
        if deployment_manifest.get(field_name) != metadata.get(field_name):
            raise InferenceError(
                f"Checkpoint and deployment manifest disagree on {field_name}."
            )
    if deployment_manifest.get("labels", {}).get("species") != class_names:
        raise InferenceError("Checkpoint and deployment manifest disagree on species order.")
    heads = deployment_manifest.get("heads")
    if not isinstance(heads, dict):
        raise InferenceError("The v2 deployment manifest has no head schema.")
    for head_name, expected_count in (
        ("species", len(class_names)),
        ("sex", 2),
        ("mating", 2),
        ("morph", 2),
    ):
        schema = heads.get(head_name)
        if not isinstance(schema, dict) or not isinstance(schema.get("output_key"), str):
            raise InferenceError(f"The v2 {head_name} head schema is invalid.")
        if head_name == "species":
            schema_labels = class_names
        else:
            schema_labels = schema.get("class_names")
            if not isinstance(schema_labels, list) or not all(
                isinstance(item, str) for item in schema_labels
            ):
                raise InferenceError(f"The v2 {head_name} head labels are invalid.")
        classifier_weight = state_dict.get(f"{head_name}_head.weight")
        if classifier_weight is None or int(classifier_weight.shape[0]) != expected_count:
            raise InferenceError(f"The v2 {head_name} head size does not match its schema.")
        if len(schema_labels) != expected_count:
            raise InferenceError(f"The v2 {head_name} label count does not match its head.")
    try:
        model = build_model(
            str(metadata["backbone"]),
            len(class_names),
            pretrained=False,
            dropout=float(metadata.get("training_config", {}).get("dropout", 0.2)),
        ).to(device)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise InferenceError("Failed to build or load the v2 multi-task model.") from exc
    model.eval()
    image_size = int(metadata.get("image_size", 384))
    crop_mode = str(metadata.get("crop_mode", "letterbox"))
    preprocess = build_transform(
        image_size=image_size,
        training=False,
        crop_mode=crop_mode,
        normalization_mean=metadata.get("normalization_mean", DEFAULT_IMAGENET_MEAN),
        normalization_std=metadata.get("normalization_std", DEFAULT_IMAGENET_STD),
    )
    config = dict(metadata)
    config.setdefault("model_name", f"{metadata.get('backbone', 'unknown')}_multitask")
    config["num_classes"] = len(class_names)
    config["heads"] = heads
    deployment_thresholds = deployment_manifest.get("thresholds")
    if not isinstance(deployment_thresholds, dict):
        raise InferenceError("The v2 deployment manifest has no threshold schema.")
    if deployment_thresholds.get("species_single") != metadata.get("thresholds", {}).get(
        "species_single"
    ):
        raise InferenceError(
            "Checkpoint and deployment manifest disagree on the single-image threshold."
        )
    config["thresholds"] = deployment_thresholds
    inference_config = deployment_manifest.get("inference")
    if not isinstance(inference_config, dict):
        inference_config = {}
    # The locked validation and Capstone reports use horizontal-flip test-time
    # augmentation, so production inference follows the same protocol.
    inference_config.setdefault("horizontal_flip_tta", True)
    config["inference"] = inference_config
    detector_models = None
    detector_config = inference_config.get("detector_preprocess")
    if isinstance(detector_config, dict) and detector_config.get("enabled", False):
        detector_files = detector_config.get("model_files")
        if not isinstance(detector_files, list) or len(detector_files) != 2:
            raise InferenceError("Detector preprocessing must name exactly two model files.")
        detector_paths = [checkpoint_path.with_name(name) for name in detector_files]
        if not all(path.is_file() for path in detector_paths):
            missing = [str(path) for path in detector_paths if not path.is_file()]
            raise InferenceError(f"Detector checkpoint is missing: {', '.join(missing)}")
        try:
            from ultralytics import YOLO

            detector_models = (YOLO(str(detector_paths[0])), YOLO(str(detector_paths[1])))
        except Exception as exc:
            raise InferenceError("Failed to load the dual-detector preprocessing models.") from exc
    return ModelArtifacts(
        model=model,
        preprocess=preprocess,
        class_names=list(class_names),
        config=config,
        device=device,
        artifact_version=2,
        labels=labels,
        detector_models=detector_models,
    )


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection *= max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return float(intersection / max(1e-9, first_area + second_area - intersection))


def _fit_pad_square(image: Image.Image, size: int, fill: Sequence[int]) -> Image.Image:
    contained = ImageOps.contain(image, (size, size), Image.Resampling.LANCZOS)
    square = Image.new("RGB", (size, size), tuple(int(value) for value in fill))
    square.paste(contained, ((size - contained.width) // 2, (size - contained.height) // 2))
    return square


def prepare_classifier_image(
    image: Image.Image, artifacts: ModelArtifacts
) -> tuple[Image.Image, Dict[str, Any]]:
    """Apply the conservative dual-detector crop used to build the 512px dataset."""
    detector_config = artifacts.config.get("inference", {}).get("detector_preprocess")
    if not isinstance(detector_config, dict) or not detector_config.get("enabled", False):
        return image, {"mode": "model_transform_only"}
    if artifacts.detector_models is None:
        raise InferenceError("Detector preprocessing is enabled but its models are unavailable.")

    max_side = int(detector_config.get("max_side", 1600))
    working = image.copy()
    working.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    confidence_threshold = float(detector_config.get("confidence_threshold", 0.25))
    agreement_threshold = float(detector_config.get("agreement_iou_threshold", 0.70))
    margin_ratio = float(detector_config.get("margin_ratio", 0.15))
    sizes = detector_config.get("inference_sizes", [512, 1024])
    detector_device: Any = 0 if artifacts.device.type == "cuda" else "cpu"
    try:
        results = [
            model.predict(
                np.asarray(working), imgsz=int(size), conf=0.01,
                device=detector_device, verbose=False,
            )[0]
            for model, size in zip(artifacts.detector_models, sizes)
        ]
    except Exception as exc:
        raise InferenceError("Object detection failed during image preprocessing.") from exc

    fallback = not len(results[0].boxes) or not len(results[1].boxes)
    confidence_512 = confidence_1024 = agreement = 0.0
    first_box = second_box = None
    if not fallback:
        first_index = int(results[0].boxes.conf.argmax())
        second_index = int(results[1].boxes.conf.argmax())
        first_box = results[0].boxes.xyxy[first_index].cpu().numpy()
        second_box = results[1].boxes.xyxy[second_index].cpu().numpy()
        confidence_512 = float(results[0].boxes.conf[first_index])
        confidence_1024 = float(results[1].boxes.conf[second_index])
        agreement = _box_iou(first_box, second_box)
        fallback = (
            min(confidence_512, confidence_1024) < confidence_threshold
            or agreement < agreement_threshold
        )

    if fallback:
        selected = working
        crop_fraction = 1.0
        mode = "fallback_original"
    else:
        union = np.array(
            [
                min(first_box[0], second_box[0]), min(first_box[1], second_box[1]),
                max(first_box[2], second_box[2]), max(first_box[3], second_box[3]),
            ]
        )
        padding = max(union[2] - union[0], union[3] - union[1]) * margin_ratio
        crop_box = (
            max(0, round(union[0] - padding)), max(0, round(union[1] - padding)),
            min(working.width, round(union[2] + padding)),
            min(working.height, round(union[3] + padding)),
        )
        selected = working.crop(crop_box)
        crop_fraction = selected.width * selected.height / (working.width * working.height)
        mode = "dual_agreement_crop"

    output_size = int(detector_config.get("output_size", 512))
    prepared = _fit_pad_square(selected, output_size, detector_config.get("pad_fill", [124, 116, 104]))
    return prepared, {
        "mode": mode,
        "confidence_512": confidence_512,
        "confidence_1024": confidence_1024,
        "agreement_iou": agreement,
        "crop_area_fraction": crop_fraction,
        "output_size": [output_size, output_size],
    }


def load_model_artifacts(
    checkpoint_path: Path = CHECKPOINT_PATH,
    class_names_path: Path = CLASS_NAMES_PATH,
    model_config_path: Path = MODEL_CONFIG_PATH,
    device_name: str = "auto",
    *,
    use_active_model: bool = True,
) -> ModelArtifacts:
    """Load model, preprocessing, class names, and configuration for inference."""
    device = _resolve_device(device_name)
    using_default_paths = (
        checkpoint_path == CHECKPOINT_PATH
        and class_names_path == CLASS_NAMES_PATH
        and model_config_path == MODEL_CONFIG_PATH
    )
    if use_active_model and using_default_paths and ACTIVE_MODEL_PATH.is_file():
        active = _read_json(ACTIVE_MODEL_PATH)
        if not isinstance(active, dict):
            raise InferenceError("active_model.json must contain an object.")
        artifact_version = int(active.get("artifact_version", 1))
        selected = active.get("checkpoint")
        if not isinstance(selected, str) or not selected:
            raise InferenceError("active_model.json must name a checkpoint.")
        selected_path = (MODEL_DIR / selected).resolve()
        try:
            selected_path.relative_to(MODEL_DIR.resolve())
        except ValueError as exc:
            raise InferenceError("Active checkpoint must stay inside the model directory.") from exc
        if artifact_version == 2:
            return _load_v2_artifacts(selected_path, device)
        if artifact_version != 1:
            raise InferenceError(f"Unsupported active artifact version: {artifact_version}")
        checkpoint_path = selected_path
    class_names = _read_json(class_names_path)
    model_config = _read_json(model_config_path)

    if not isinstance(class_names, list) or not all(isinstance(name, str) for name in class_names):
        raise InferenceError("class_names.json must contain a JSON list of class-name strings.")
    if not isinstance(model_config, dict):
        raise InferenceError("model_config.json must contain a JSON object.")
    if len(class_names) != int(model_config.get("num_classes", len(class_names))):
        raise InferenceError("Class-name count does not match model_config.json num_classes.")

    model_name = model_config.get("model_name", SUPPORTED_MODEL_NAME)
    if model_name != SUPPORTED_MODEL_NAME:
        raise InferenceError(f"Unsupported model architecture in config: {model_name}")

    checkpoint = _load_checkpoint(checkpoint_path, device)
    checkpoint_model_name = checkpoint.get("model_name", SUPPORTED_MODEL_NAME)
    if checkpoint_model_name != SUPPORTED_MODEL_NAME:
        raise InferenceError(f"Unsupported model architecture in checkpoint: {checkpoint_model_name}")

    checkpoint_class_names = checkpoint.get("class_names")
    if isinstance(checkpoint_class_names, list) and checkpoint_class_names != class_names:
        raise InferenceError("Checkpoint class order does not match class_names.json.")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise InferenceError("Checkpoint does not contain model_state_dict.")

    classifier_weight = state_dict.get("classifier.1.weight")
    if classifier_weight is None or classifier_weight.shape[0] != len(class_names):
        raise InferenceError("Checkpoint output layer does not match the loaded class count.")

    model = _build_mobilenet_v2(num_classes=len(class_names)).to(device)
    try:
        model.load_state_dict(state_dict)
    except Exception as exc:
        raise InferenceError("Failed to load model weights into MobileNetV2.") from exc
    model.eval()

    return ModelArtifacts(
        model=model,
        preprocess=_build_preprocess(model_config),
        class_names=list(class_names),
        config=model_config,
        device=device,
        artifact_version=1,
    )


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Decode uploaded image bytes and return an RGB Pillow image."""
    if not image_bytes:
        raise InferenceError("The uploaded image is empty.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InferenceError("The uploaded file is not a valid readable image.") from exc
    except Exception as exc:
        raise InferenceError("Could not open the uploaded image.") from exc


def top_k_predictions(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Return top-k class probabilities in descending order."""
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    if probabilities_array.ndim != 1 or probabilities_array.size != len(class_names):
        raise InferenceError("Probability vector length does not match class names.")

    top_count = min(k, len(class_names))
    top_indices = np.argsort(probabilities_array)[::-1][:top_count]
    return [
        {
            "rank": rank,
            "class_name": class_names[int(class_index)],
            "class_index": int(class_index),
            "probability": float(probabilities_array[int(class_index)]),
        }
        for rank, class_index in enumerate(top_indices, start=1)
    ]


def summarize_probabilities(
    probabilities: Sequence[float],
    class_names: Sequence[str],
    k: int = 5,
) -> Dict[str, Any]:
    """Build a prediction summary from a probability vector."""
    probabilities_array = np.asarray(probabilities, dtype=np.float64)
    top_predictions = top_k_predictions(probabilities_array, class_names=class_names, k=k)
    best = top_predictions[0]
    return {
        "prediction": best["class_name"],
        "class_index": best["class_index"],
        "confidence": best["probability"],
        "probabilities": probabilities_array,
        "top_k": top_predictions,
    }


def enrich_species_summary(
    summary: Dict[str, Any], artifacts: ModelArtifacts
) -> Dict[str, Any]:
    """Attach structured taxonomy to a primary-head summary when available."""

    enriched = dict(summary)
    if artifacts.artifact_version == 2:
        species_name = str(enriched["prediction"])
        enriched["family"] = str(artifacts.labels["species_to_family"][species_name])
        enriched["species"] = {
            "prediction": species_name,
            "class_index": int(enriched["class_index"]),
            "confidence": float(enriched["confidence"]),
            "probabilities": enriched["probabilities"],
            "top_k": enriched["top_k"],
        }
    return enriched


def predict_image_bytes(
    image_bytes: bytes,
    artifacts: ModelArtifacts,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run model inference for uploaded image bytes."""
    image = load_image_from_bytes(image_bytes)
    prepared_image, preprocessing = prepare_classifier_image(image, artifacts)
    try:
        image_tensor = artifacts.preprocess(prepared_image).unsqueeze(0).to(artifacts.device)
        with torch.inference_mode():
            outputs = artifacts.model(image_tensor)
            if isinstance(outputs, dict):
                heads = artifacts.config["heads"]
                species_output_key = heads["species"]["output_key"]
                sex_output_key = heads["sex"]["output_key"]
                mating_output_key = heads["mating"]["output_key"]
                morph_output_key = heads["morph"]["output_key"]
                temperature = float(
                    artifacts.config.get("calibration", {}).get("species_temperature", 1.0)
                )
                if temperature <= 0:
                    raise InferenceError("Species calibration temperature must be positive.")
                probabilities_tensor = torch.softmax(
                    outputs[species_output_key] / temperature, dim=1
                )
                sex_tensor = torch.softmax(outputs[sex_output_key], dim=1)
                mating_tensor = torch.softmax(outputs[mating_output_key], dim=1)
                morph_tensor = torch.softmax(outputs[morph_output_key], dim=1)
                if bool(
                    artifacts.config.get("inference", {}).get(
                        "horizontal_flip_tta", False
                    )
                ):
                    flipped = artifacts.model(torch.flip(image_tensor, dims=(3,)))
                    probabilities_tensor = (
                        probabilities_tensor
                        + torch.softmax(flipped[species_output_key] / temperature, dim=1)
                    ) / 2
                    sex_tensor = (
                        sex_tensor + torch.softmax(flipped[sex_output_key], dim=1)
                    ) / 2
                    mating_tensor = (
                        mating_tensor + torch.softmax(flipped[mating_output_key], dim=1)
                    ) / 2
                    morph_tensor = (
                        morph_tensor + torch.softmax(flipped[morph_output_key], dim=1)
                    ) / 2
                probabilities = probabilities_tensor.squeeze(0).cpu().numpy()
                sex_probabilities = sex_tensor.squeeze(0).cpu().numpy()
                mating_probabilities = mating_tensor.squeeze(0).cpu().numpy()
                morph_probabilities = morph_tensor.squeeze(0).cpu().numpy()
            else:
                probabilities = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                sex_probabilities = mating_probabilities = morph_probabilities = None
    except Exception as exc:
        raise InferenceError("Prediction failed while running the model.") from exc

    if probabilities.shape[0] != len(artifacts.class_names):
        raise InferenceError("Model output length does not match the loaded class list.")

    summary = enrich_species_summary(
        summarize_probabilities(probabilities, artifacts.class_names, k=top_k), artifacts
    )
    summary["preprocessing"] = preprocessing
    if artifacts.artifact_version != 2:
        return summary

    species_name = str(summary["prediction"])
    sex_labels = list(artifacts.config["heads"]["sex"]["class_names"])
    mating_labels = list(artifacts.config["heads"]["mating"]["class_names"])
    morph_labels = list(artifacts.config["heads"]["morph"]["class_names"])
    try:
        mating_index = mating_labels.index("mating")
        not_mating_index = mating_labels.index("not_mating")
    except ValueError as exc:
        raise InferenceError(
            "The mating head schema must define mating and not_mating labels."
        ) from exc
    sex_index = int(np.argmax(sex_probabilities))
    mating_probability = float(mating_probabilities[mating_index])
    not_mating_probability = float(mating_probabilities[not_mating_index])
    sex_name = sex_labels[sex_index]
    observation_labels = [*sex_labels, "mating"]
    observation_probabilities = np.concatenate(
        [sex_probabilities * not_mating_probability, [mating_probability]]
    )
    observation_index = int(np.argmax(observation_probabilities))
    observation_name = observation_labels[observation_index]
    observation_confidence = float(observation_probabilities[observation_index])
    morph_index = int(np.argmax(morph_probabilities))
    morph_name = morph_labels[morph_index]
    morph_species = artifacts.config["heads"]["morph"].get(
        "applicable_species", []
    )
    summary.update(
        {
            "observation_state": {
                "prediction": observation_name,
                "confidence": observation_confidence,
                "probabilities": {
                    label: float(observation_probabilities[index])
                    for index, label in enumerate(observation_labels)
                },
                "sex_prediction": sex_name,
                "sex_confidence": float(sex_probabilities[sex_index]),
                "mating_probability": mating_probability,
            },
            "morph": (
                {
                    "prediction": morph_name,
                    "class_index": morph_index,
                    "confidence": float(morph_probabilities[morph_index]),
                    "probabilities": {
                        label: float(morph_probabilities[index])
                        for index, label in enumerate(morph_labels)
                    },
                }
                if species_name in morph_species
                else None
            ),
        }
    )
    return summary


def combine_same_specimen_probabilities(
    probabilities_image_1: Sequence[float],
    probabilities_image_2: Sequence[float],
) -> np.ndarray:
    """Combine two views under the assumption that they show the same specimen.

    The per-class scores are combined with an equal-weight logarithmic pool
    (a normalized geometric mean). This is a conservative same-class
    joint-evidence rule: a class must be supported by both views, rather than
    either view being allowed to dominate through arithmetic averaging.
    """
    first = np.asarray(probabilities_image_1, dtype=np.float64)
    second = np.asarray(probabilities_image_2, dtype=np.float64)
    if first.shape != second.shape:
        raise InferenceError("Cannot combine probability vectors with different shapes.")
    if first.ndim != 1 or first.size == 0:
        raise InferenceError("Probability vectors must be one-dimensional.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise InferenceError("Probability vectors must contain only finite values.")
    if np.any(first < 0.0) or np.any(second < 0.0):
        raise InferenceError("Probability vectors cannot contain negative values.")

    first_total = float(first.sum())
    second_total = float(second.sum())
    if first_total <= 0.0 or second_total <= 0.0:
        raise InferenceError("Probability vectors must each have a positive sum.")

    first = first / first_total
    second = second / second_total
    if not np.any((first > 0.0) & (second > 0.0)):
        raise InferenceError("The two images have no overlapping class support.")

    # Work in log space so very small softmax probabilities do not underflow.
    smallest_positive = np.finfo(np.float64).tiny
    joint_log_scores = 0.5 * np.log(np.clip(first, smallest_positive, None))
    joint_log_scores += 0.5 * np.log(np.clip(second, smallest_positive, None))
    joint_log_scores -= float(np.max(joint_log_scores))
    joint_scores = np.exp(joint_log_scores)
    return joint_scores / float(joint_scores.sum())


def combine_probabilities(
    probabilities_image_1: Sequence[float],
    probabilities_image_2: Sequence[float],
) -> np.ndarray:
    """Backward-compatible alias for same-specimen probability fusion."""
    return combine_same_specimen_probabilities(
        probabilities_image_1,
        probabilities_image_2,
    )


def combine_within_candidate_classes(
    probabilities_image_1: Sequence[float],
    probabilities_image_2: Sequence[float],
    candidate_indices: Sequence[int],
) -> np.ndarray:
    """Fuse two views while restricting the result to Image 1 candidates."""
    first = np.asarray(probabilities_image_1, dtype=np.float64)
    second = np.asarray(probabilities_image_2, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise InferenceError("Cannot restrict probability vectors with different shapes.")
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if candidates.ndim != 1 or candidates.size == 0:
        raise InferenceError("At least one candidate class is required.")
    if np.unique(candidates).size != candidates.size:
        raise InferenceError("Candidate classes must be unique.")
    if np.any(candidates < 0) or np.any(candidates >= first.size):
        raise InferenceError("A candidate class index is outside the model output.")

    restricted_joint = combine_same_specimen_probabilities(
        first[candidates], second[candidates]
    )
    result = np.zeros_like(first, dtype=np.float64)
    result[candidates] = restricted_joint
    return result


def restrict_probabilities_to_candidates(
    probabilities: Sequence[float], candidate_indices: Sequence[int]
) -> np.ndarray:
    """Condition a probability vector on a fixed set of candidate classes."""
    values = np.asarray(probabilities, dtype=np.float64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if values.ndim != 1 or candidates.ndim != 1 or candidates.size == 0:
        raise InferenceError("A one-dimensional probability vector and candidates are required.")
    if np.unique(candidates).size != candidates.size:
        raise InferenceError("Candidate classes must be unique.")
    if np.any(candidates < 0) or np.any(candidates >= values.size):
        raise InferenceError("A candidate class index is outside the model output.")
    restricted = np.zeros_like(values)
    restricted[candidates] = values[candidates]
    total = float(restricted.sum())
    if total <= 0.0:
        raise InferenceError("The image has no probability support for the candidates.")
    return restricted / total
