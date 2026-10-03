"""Streamlit app for the packaged caddisfly image classifier."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import traceback
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from inference import (
    ACTIVE_MODEL_PATH,
    InferenceError,
    combine_within_candidate_classes,
    enrich_species_summary,
    load_image_from_bytes,
    load_model_artifacts,
    predict_image_bytes,
    restrict_probabilities_to_candidates,
    summarize_probabilities,
)


CONFIDENCE_THRESHOLD = 0.70
PACKAGE_ROOT = Path(__file__).resolve().parent
SPECIES_PROFILES_PATH = PACKAGE_ROOT / "web_assets" / "species" / "profiles.json"
FEEDBACK_DIR = PACKAGE_ROOT / "user_feedback"
SUPPORTED_UPLOAD_TYPES = ("jpg", "jpeg", "png", "webp")
SESSION_KEYS = (
    "first_image_bytes",
    "first_image_name",
    "first_prediction",
    "second_image_bytes",
    "second_image_name",
    "second_prediction",
    "combined_prediction",
    "same_specimen_confirmed",
    "pair_nonce",
)


st.set_page_config(
    page_title="Caddisfly Image Classifier",
    page_icon="C",
    layout="wide",
)


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1320px;
            padding-top: 1.6rem;
            padding-bottom: 3.5rem;
        }
        h1, h2, h3, p {
            letter-spacing: 0;
        }
        .app-title {
            color: #24332C;
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.15;
            margin: 0 0 0.35rem;
        }
        .app-subtitle {
            color: #53645A;
            font-size: 1rem;
            line-height: 1.45;
            margin: 0 0 0.75rem;
        }
        .result-card {
            border: 1px solid #DDE4DA;
            border-radius: 18px;
            padding: 1.25rem 1.35rem 1.3rem;
            background: linear-gradient(145deg, #ffffff 0%, #F7F8F4 100%);
            margin: 0 0 0.75rem;
            box-shadow: 0 10px 30px rgba(23, 58, 94, 0.08);
        }
        .result-label {
            color: #245B45;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
        }
        .result-title {
            color: #24332C;
            font-size: 1.55rem;
            font-style: italic;
            font-weight: 760;
            line-height: 1.35;
            margin-bottom: 0.7rem;
            overflow-wrap: anywhere;
        }
        .field-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.45rem;
            margin: 0.75rem 0;
        }
        .field-box {
            background: #F0F3EC;
            border: 1px solid #DDE4DA;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
        }
        .field-name {
            color: #68766B;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }
        .field-value {
            color: #24332C;
            font-size: 0.92rem;
            margin-top: 0.1rem;
            overflow-wrap: anywhere;
        }
        .confidence-line {
            color: #24332C;
            font-size: 1rem;
            margin-top: 0.35rem;
        }
        .status-pill {
            display: inline-block;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 700;
            padding: 0.25rem 0.6rem;
            margin-top: 0.55rem;
        }
        .status-good {
            background: rgba(46, 125, 91, 0.12);
            color: #2E7D5B;
        }
        .status-warn {
            background: rgba(183, 121, 31, 0.14);
            color: #B7791F;
        }
        .interpretation {
            color: #53645A;
            font-size: 0.92rem;
            line-height: 1.45;
            margin-top: 0.6rem;
        }
        .image-frame {
            border: 1px solid #DDE4DA;
            border-radius: 18px;
            background: #F0F3EC;
            padding: 0.65rem;
            margin-bottom: 0.75rem;
            text-align: center;
        }
        .image-frame img {
            max-height: 460px;
            width: 100%;
            object-fit: contain;
            border-radius: 8px;
            display: block;
            margin: 0 auto;
        }
        .image-frame.compact img {
            max-height: 230px;
        }
        .image-caption {
            color: #68766B;
            font-size: 0.82rem;
            margin-top: 0.45rem;
            overflow-wrap: anywhere;
        }
        .top5-list {
            display: grid;
            gap: 0.72rem;
            margin: 0.2rem 0 0.35rem;
        }
        .top5-row {
            border: 1px solid #DDE4DA;
            border-radius: 8px;
            background: #ffffff;
            padding: 0.65rem 0.75rem;
        }
        .top5-meta {
            display: grid;
            grid-template-columns: 2rem minmax(0, 1fr) 5.5rem;
            gap: 0.55rem;
            align-items: start;
            color: #24332C;
            font-size: 0.9rem;
        }
        .top5-rank {
            color: #245B45;
            font-weight: 760;
        }
        .top5-name {
            overflow-wrap: anywhere;
            line-height: 1.35;
        }
        .top5-prob {
            color: #24332C;
            font-weight: 760;
            text-align: right;
        }
        .prob-track {
            height: 0.42rem;
            border-radius: 999px;
            background: #E5ECF1;
            overflow: hidden;
            margin-top: 0.45rem;
        }
        .prob-fill {
            height: 100%;
            border-radius: 999px;
            background: #245B45;
        }
        .compact-heading {
            color: #24332C;
            font-size: 1.25rem;
            font-weight: 740;
            margin: 0.5rem 0 0.15rem;
        }
        .helper-text {
            color: #53645A;
            font-size: 0.94rem;
            line-height: 1.45;
            margin-bottom: 0.7rem;
        }
        .mini-result {
            border: 1px solid #DDE4DA;
            border-radius: 9px;
            background: #ffffff;
            padding: 0.75rem 0.85rem;
            margin-bottom: 0.75rem;
        }
        .mini-title {
            color: #68766B;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .mini-label {
            color: #24332C;
            font-weight: 720;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .mini-confidence {
            color: #53645A;
            margin-top: 0.35rem;
            font-size: 0.92rem;
        }
        .footer-note {
            color: #68766B;
            border-top: 1px solid #DDE4DA;
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 1.25rem;
            padding-top: 0.8rem;
        }
        .profile-hero {
            margin-top: 1.35rem;
            padding: 1.1rem 1.25rem;
            border-radius: 16px;
            color: #ffffff;
            background: linear-gradient(115deg, #245B45 0%, #36735A 72%, #245B45 100%);
            box-shadow: 0 12px 32px rgba(23, 58, 94, 0.14);
        }
        .profile-kicker {
            color: #F7B267;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .profile-name {
            margin-top: 0.18rem;
            font-size: 1.65rem;
            font-weight: 760;
            font-style: italic;
        }
        .profile-family {
            margin-top: 0.18rem;
            color: #DCEAF2;
            font-size: 0.94rem;
        }
        .profile-caution {
            margin: 0.75rem 0 0.35rem;
            padding: 0.72rem 0.85rem;
            border-left: 4px solid #F28C28;
            border-radius: 0 10px 10px 0;
            background: #FFF7ED;
            color: #7C4214;
            line-height: 1.45;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            margin-top: 0.65rem;
            padding: 0.3rem;
            border-radius: 13px;
            background: #EEF4F7;
        }
        button[data-baseweb="tab"] {
            border-radius: 10px;
            padding-left: 1.15rem;
            padding-right: 1.15rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: #ffffff;
            color: #245B45;
            box-shadow: 0 2px 8px rgba(23, 58, 94, 0.10);
        }
        div[data-testid="stMetric"] {
            padding: 0.85rem 0.95rem;
            border: 1px solid #DFE8ED;
            border-radius: 13px;
            background: #F7F8F4;
        }
        @media (max-width: 760px) {
            .field-grid { grid-template-columns: 1fr; }
            .top5-meta {
                grid-template-columns: 1.6rem minmax(0, 1fr);
            }
            .top5-prob {
                grid-column: 2;
                text-align: left;
            }
            .app-title {
                font-size: 1.7rem;
            }
        }
        .stApp { background: #F8F5ED; color: #283B31; }
        .block-container { max-width: 1180px; padding-top: 2.1rem; padding-bottom: 3rem; }
        h1, h2, h3 { font-family: Georgia, "Times New Roman", serif !important; font-weight: 400 !important; color: #283B31; }
        h1 { font-size: clamp(2.1rem, 4vw, 3.3rem) !important; letter-spacing: -.035em !important; }
        h3 { font-size: 1.65rem !important; }
        hr { border-color: #DCDDCF !important; margin: 1.8rem 0 !important; }
        .site-brand { font-family: Georgia, serif; color: #283B31; font-size: 1.35rem; letter-spacing: .015em; padding-top: .3rem; }
        .site-brand span { display: block; font-family: sans-serif; font-size: .65rem; letter-spacing: .2em; color: #7A8270; margin-top: .25rem; text-transform: uppercase; }
        .hero-kicker, .section-kicker { color: #788367; text-transform: uppercase; letter-spacing: .2em; font-size: .7rem; margin-top: 2.7rem; }
        h1.hero-title { font-family: Georgia, "Times New Roman", serif !important; font-size: clamp(2.8rem, 5.5vw, 4.8rem) !important; line-height: 1.06; letter-spacing: -.055em !important; margin: 1.3rem 0 1.4rem; font-weight: 400 !important; }
        .hero-title em { color: #6C8062; font-weight: 400; }
        .hero-description { color: #687063; font-size: 1rem; line-height: 1.9; max-width: 390px; margin-bottom: 1.8rem; }
        .specimen-plate { position: relative; background: #E9EBDD; padding: 1.4rem; border-radius: 18px; margin: .8rem 0 1rem; }
        .specimen-plate img { display: block; width: 100%; height: 460px; object-fit: contain; border-radius: 10px; background: #12150E; }
        .plate-label { display: flex; justify-content: space-between; gap: 1rem; border-top: 1px solid #C8CFBD; padding-top: .85rem; margin-top: 1.3rem; font-size: .67rem; letter-spacing: .13em; color: #6E7863; text-transform: uppercase; }
        .plate-species { font-family: Georgia, serif; font-size: 1.1rem; font-style: italic; letter-spacing: 0; text-transform: none; color: #354B3B; }
        .editorial-note { padding: .4rem 0 1rem; }
        .editorial-note .section-kicker { margin-top: 0; color: #9A8160; }
        .editorial-note h3 { margin: .65rem 0; }
        .editorial-note p { color: #687063; font-size: .95rem; line-height: 1.85; max-width: 450px; }
        [data-testid="stButton"] button { border-radius: 999px; border: 1px solid #CFD6C7; background: transparent; padding: .65rem 1.3rem; transition: background .18s ease, border-color .18s ease; }
        [data-testid="stButton"] button:hover { background: #E8ECDD; border-color: #8E9D81; color: #283B31; }
        [data-testid="stButton"] button[kind="primary"] { background: #304C3B; color: #FAF8F1; border-color: #304C3B; }
        [data-testid="stButton"] button[kind="primary"]:hover { background: #43624B; border-color: #43624B; }
        [data-testid="stButton"] button:focus-visible { outline: 3px solid #A9BA94; outline-offset: 3px; }
        [data-testid="stFileUploader"] { border: 1px dashed #B2BEA5; border-radius: 18px; padding: 1.4rem; background: #F0F1E7; }
        [data-testid="stFileUploaderDropzone"] { background: transparent; }
        .image-frame { border: 0; padding: .8rem; background: #EDEFE3; border-radius: 16px; }
        .image-frame img { max-height: 500px; border-radius: 10px; }
        .image-caption { color: #7D8575; font-size: .75rem; padding: .4rem; }
        .result-card { border: 0; border-top: 1px solid #BBC6B1; border-radius: 0; box-shadow: none; background: transparent; padding: 1.4rem 0; }
        .result-label { color: #8B785B; letter-spacing: .16em; font-size: .68rem; }
        .result-title { font-family: Georgia, serif; color: #304C3B; font-weight: 400; font-size: 2.15rem; margin: .6rem 0 1.2rem; }
        .field-box { border: 0; border-left: 1px solid #D5DDCC; background: transparent; border-radius: 0; }
        .field-name { font-size: .65rem; letter-spacing: .08em; color: #828A78; }
        .confidence-line { font-size: .9rem; }
        .status-pill { font-size: .75rem; padding: .35rem .8rem; }
        .interpretation { color: #74806C; font-size: .85rem; line-height: 1.8; }
        .profile-hero { background: transparent; color: #304C3B; border-radius: 0; border-top: 1px solid #CDD6C3; box-shadow: none; padding: 1.7rem 0 .7rem; margin-top: 2rem; }
        .profile-kicker { color: #8B785B; font-size: .65rem; font-weight: 500; letter-spacing: .18em; }
        .profile-name { font-family: Georgia, serif; font-size: 2.2rem; font-weight: 400; margin: .55rem 0; }
        .profile-family { color: #74806C; }
        .profile-caution { background: #F0EBDD; color: #7B6A50; border-left: 2px solid #BBA783; font-size: .85rem; }
        div[data-testid="stMetric"] { background: transparent; border: 0; border-bottom: 1px solid #D5DDCC; border-radius: 0; padding: .8rem 0; }
        .top5-row, .mini-result { background: #F3F3E9; border-color: #D8DECF; box-shadow: none; }
        .footer-note { font-size: .75rem; color: #858C7B; border-color: #DCDDCF; margin-top: 2.5rem; padding-top: 1.2rem; }
        @media (max-width: 760px) {
            .block-container { padding-top: 1rem; }
            .st-key-site_navigation [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .4rem; }
            .st-key-site_navigation [data-testid="stColumn"] { min-width: 0 !important; }
            .st-key-site_navigation [data-testid="stColumn"]:first-child { flex: 2 !important; }
            .st-key-site_navigation [data-testid="stColumn"]:not(:first-child) { flex: 1 !important; }
            .st-key-site_navigation button { padding: .45rem .65rem; }
            .site-brand { font-size: 1.1rem; }
            .site-brand span { font-size: .55rem; }
            .hero-kicker { margin-top: .8rem; }
            h1.hero-title { font-size: 3rem !important; }
            .specimen-plate img { height: 380px; }
            .plate-label { flex-direction: column; gap: .35rem; }
            .result-title { font-size: 1.8rem; }
            [data-testid="stFileUploader"] { padding: .8rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def active_model_signature() -> Tuple[int, int] | None:
    """Return a cache key that changes whenever the production selector changes."""

    if not ACTIVE_MODEL_PATH.is_file():
        return None
    stat = ACTIVE_MODEL_PATH.stat()
    return stat.st_mtime_ns, stat.st_size


@st.cache_resource(show_spinner="Loading the active caddisfly model...")
def get_model_artifacts(active_signature: Tuple[int, int] | None = None):
    del active_signature
    if st.session_state.get("cloud_demo", False):
        from model_assets import ensure_model_assets

        release_url = st.secrets.get("deployment", {}).get("release_base_url", "")
        ensure_model_assets(PACKAGE_ROOT, release_url)
        return load_model_artifacts(device_name="cpu")
    return load_model_artifacts(device_name="auto")


def initialise_session_state() -> None:
    st.session_state.setdefault("page", "home")
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0
    for key in SESSION_KEYS:
        if key == "same_specimen_confirmed":
            default_value = False
        elif key == "pair_nonce":
            default_value = 0
        else:
            default_value = None
        st.session_state.setdefault(key, default_value)


def reset_application() -> None:
    next_nonce = int(st.session_state.get("uploader_nonce", 0)) + 1
    for key in list(st.session_state.keys()):
        if (
            key in SESSION_KEYS
            or key.startswith("first_uploader_")
            or key.startswith("second_uploader_")
        ):
            del st.session_state[key]
    st.session_state.uploader_nonce = next_nonce
    st.session_state.page = "identify"


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def artifact_confidence_threshold(artifacts) -> float:
    return float(
        artifacts.config.get("thresholds", {}).get(
            "species_single", CONFIDENCE_THRESHOLD
        )
    )


def artifact_joint_confidence_threshold(artifacts) -> float | None:
    """Return a validated two-view threshold, if the artifact defines one."""

    value = artifacts.config.get("thresholds", {}).get("species_joint")
    return None if value is None else float(value)


def requires_second_image(
    confidence: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """Return whether the first result must enter the second-image workflow."""
    return float(confidence) < float(threshold)


def joint_result_is_accepted(
    first_label: str,
    second_label: str,
    combined_confidence: float,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> bool:
    """Accept a pair only when both views agree and joint confidence passes."""
    return (
        first_label == second_label
        and float(combined_confidence) >= float(threshold)
    )


def parse_class_label(label: str) -> Tuple[str, str, str]:
    sex = "Not parsed"
    core = label
    for candidate in (" female", " male"):
        if label.endswith(candidate):
            sex = candidate.strip()
            core = label[: -len(candidate)].strip()
            break

    parts = core.split(maxsplit=1)
    if not parts:
        return "Not parsed", "Not parsed", sex
    family = parts[0].rstrip(",")
    taxon = parts[1] if len(parts) > 1 else "Not parsed"
    return family, taxon, sex


def show_top3(top_predictions: List[Dict[str, Any]], title: str, expanded: bool = True) -> None:
    html_parts = ['<div class="top5-list">']
    for item in top_predictions:
        probability = float(item["probability"])
        width = max(0.0, min(probability * 100.0, 100.0))
        html_parts.append(
            '<div class="top5-row">'
            '<div class="top5-meta">'
            f'<div class="top5-rank">{int(item["rank"])}</div>'
            f'<div class="top5-name">{str(item["class_name"])}</div>'
            f'<div class="top5-prob">{format_percent(probability)}</div>'
            '</div>'
            '<div class="prob-track">'
            f'<div class="prob-fill" style="width: {width:.2f}%"></div>'
            '</div>'
            '</div>'
        )
    html_parts.append("</div>")
    html_content = "".join(html_parts)

    st.markdown(f"**{title}**")
    st.markdown(html_content, unsafe_allow_html=True)


def confidence_status(confidence: float, threshold: float) -> Tuple[str, str]:
    if confidence >= threshold:
        return "Leading candidate — high model confidence", "status-good"
    return "Uncertain prediction", "status-warn"


def result_card(
    prediction: Dict[str, Any],
    threshold: float,
    heading: str = "Prediction",
    note: str | None = None,
    accepted: bool | None = None,
    status_override: Tuple[str, str] | None = None,
) -> None:
    confidence = float(prediction["confidence"])
    prediction_label = str(prediction["prediction"])
    if "family" in prediction and isinstance(prediction.get("species"), dict):
        family = str(prediction["family"])
        taxon = str(prediction["species"]["prediction"])
        observation = prediction.get("observation_state")
        if isinstance(observation, dict):
            sex = str(observation.get("prediction", "Not available"))
            observation_confidence = observation.get("confidence")
            if observation_confidence is not None:
                sex = f"{sex} ({format_percent(float(observation_confidence))})"
        else:
            sex = "Not combined"
        morph = prediction.get("morph")
        if isinstance(morph, dict):
            morph_note = (
                f"Morph: {morph.get('prediction', 'unknown')} "
                f"({format_percent(float(morph.get('confidence', 0.0)))})"
            )
            note = f"{note} {morph_note}".strip() if note else morph_note
    else:
        family, taxon, sex = parse_class_label(prediction_label)
    if status_override is not None:
        status, status_class = status_override
    elif accepted is None:
        status, status_class = confidence_status(confidence, threshold)
    elif accepted:
        status, status_class = "Leading candidate — high model confidence", "status-good"
    else:
        status, status_class = "Uncertain prediction", "status-warn"

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">{escape(heading)}</div>
            <div class="result-title">{escape(prediction_label)}</div>
            <div class="field-grid">
                <div class="field-box">
                    <div class="field-name">Family</div>
                    <div class="field-value">{escape(family)}</div>
                </div>
                <div class="field-box">
                    <div class="field-name">Species</div>
                    <div class="field-value">{escape(taxon)}</div>
                </div>
                <div class="field-box">
                    <div class="field-name">Observation</div>
                    <div class="field-value">{escape(sex)}</div>
                </div>
            </div>
            <div class="confidence-line">Confidence: <strong>{format_percent(confidence)}</strong></div>
            <div class="status-pill {status_class}">{status}</div>
            {f'<div class="interpretation">{escape(note)}</div>' if note else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_species_profiles() -> Dict[str, Any]:
    if not SPECIES_PROFILES_PATH.is_file():
        return {}
    return json.loads(SPECIES_PROFILES_PATH.read_text(encoding="utf-8"))


def profile_observation(prediction: Dict[str, Any], second_prediction=None) -> tuple[str, list[str], bool]:
    """Keep image-level sex evidence separate from the fused species score."""
    observations = [item.get("observation_state") or {} for item in
                    ([prediction, second_prediction] if second_prediction is not None else [prediction])]
    sexes = [item.get("sex_prediction") for item in observations]
    mating = any(item.get("prediction") == "mating" for item in observations)
    if len(sexes) == 2 and set(sexes) == {"male", "female"}:
        return "Sex uncertain — views disagree", ["male", "female"], mating
    if sexes and all(sex == sexes[0] for sex in sexes) and sexes[0] in ("male", "female"):
        return str(sexes[0]).capitalize(), [str(sexes[0])], mating
    return "Sex uncertain", [], mating


def matching_reference_images(profile: Dict[str, Any], sexes: list[str]) -> list[dict]:
    references = profile.get("sex_reference_images", [])
    limit = 1 if len(sexes) == 2 else 2
    return [item for sex in sexes for item in
            [reference for reference in references if reference.get("sex") == sex
             and (PACKAGE_ROOT / reference["path"]).is_file()][:limit]]


def show_species_profile(prediction: Dict[str, Any], image_bytes: bytes | None,
                         *, first_prediction=None, second_prediction=None) -> None:
    species_name = str(prediction["prediction"])
    profile = load_species_profiles().get(species_name)
    if not isinstance(profile, dict):
        return
    family = str(profile.get("family", "Not available"))
    st.markdown(
        f'<div class="profile-hero"><div class="profile-kicker">Species dossier</div>'
        f'<div class="profile-name">{escape(species_name)}</div>'
        f'<div class="profile-family">Family · {escape(family)}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="profile-caution"><strong>Use this profile to cross-check the result.</strong> '
        'Model confidence shows preference among the 25 trained classes; it does not independently verify the specimen.</div>',
        unsafe_allow_html=True,
    )
    sex_label, reference_sexes, mating = profile_observation(
        first_prediction if first_prediction is not None else prediction, second_prediction
    )
    distribution = profile.get("distribution", {})
    points = distribution.get("occurrence_points", [])
    with st.container():
        st.markdown("### Overview")
        adult_size = profile.get("adult_size")
        if isinstance(adult_size, str) and (not adult_size.strip() or "not yet verified" in adult_size.lower()):
            adult_size = None
        details = st.columns(4 if adult_size else 3, gap="medium")
        details[0].metric("Family", family)
        details[1].metric("Predicted sex", sex_label)
        details[2].metric("Mapped records", int(distribution.get("record_count", 0)))
        if adult_size:
            details[3].metric("Reported adult size", adult_size)
        if mating:
            st.markdown("**Observation state:** Mating")
        st.markdown(f"**Habitat:** {profile.get('habitat') or 'Not available.'}")
        st.markdown(
            f"**Profile status:** {profile.get('verification_status') or 'Not available.'}"
        )
        source_links = " · ".join(
            f"[{item['label']}]({item['url']})" for item in profile.get("sources", [])
        )
        st.markdown(f"**Sources:** {source_links}")
    with st.container():
        st.markdown("### Appearance")
        images = matching_reference_images(profile, reference_sexes)
        if images:
            columns = st.columns(2, gap="medium")
            for column, reference in zip(columns, images):
                path = PACKAGE_ROOT / reference["path"]
                if path.is_file():
                    column.image(
                        str(path), caption=f"{species_name} · {reference['sex'].capitalize()} · Labelled training example",
                        use_container_width=True,
                    )
            st.caption(profile.get("reference_image_note", ""))
        for sex in reference_sexes:
            if not any(item["sex"] == sex for item in images):
                st.caption(f"No labelled {sex} reference image is available for this species.")
        if not reference_sexes:
            st.caption("Sex-matched reference images require a resolved sex prediction.")
        st.markdown(
            f"**Colour and identifying features:** {profile.get('colour_and_features') or 'Species-level description not yet verified.'}"
        )
        st.caption("Natural variation, sex, lighting and viewing angle can change appearance.")
    with st.container():
        map_column, season_column = st.columns([1.45, 1], gap="large")
        with map_column:
            st.markdown("### Recorded distribution")
            if points:
                frame = pd.DataFrame(points)
                st.map(frame, latitude="latitude", longitude="longitude", color="#8A9D70", size=18)
            countries = ", ".join(distribution.get("countries", [])) or "No mapped country record"
            regions = ", ".join(distribution.get("regions", [])[:8]) or "No mapped regional record"
            st.caption(f"Countries: {countries}. Regions represented in records: {regions}.")
        with season_column:
            st.markdown("### Observation months")
            seasonality = profile.get("seasonality", {})
            if seasonality:
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                values = [int(seasonality.get(str(month), 0)) for month in range(1, 13)]
                st.bar_chart(pd.DataFrame({"records": values}, index=month_names), color="#8A9D70")
                st.caption("Observation records are sampling evidence, not a guaranteed flight season.")
            else:
                st.caption("No dated occurrence records are currently available.")
    if not st.session_state.get("cloud_demo", False):
        with st.container():
            st.markdown("### Help improve future versions")
            st.caption("Tell us which real-world traits disagree with the model's leading candidate.")
            show_feedback_form(species_name, prediction, image_bytes)


def show_feedback_form(
    species_name: str, prediction: Dict[str, Any], image_bytes: bytes | None
) -> None:
    with st.form(f"feedback_{hashlib.sha1(species_name.encode()).hexdigest()[:8]}"):
        verdict = st.radio(
            "Your assessment", ["Looks consistent", "Unsure", "Does not match"], horizontal=True
        )
        mismatches = st.multiselect(
            "What does not match?",
            ["Size", "Colour", "Location", "Season", "Body shape", "Wing pattern", "Other"],
        )
        form_left, form_right = st.columns(2)
        measured_size = form_left.text_input("Observed size (optional, include unit)")
        location = form_right.text_input("Observation location (optional)")
        expert_species = st.text_input("Alternative or expert identification (optional)")
        notes = st.text_area("Additional notes (optional)")
        allow_image = st.checkbox("Allow this uploaded image to be retained for model improvement")
        submitted = st.form_submit_button("Submit feedback", type="primary")
    if submitted:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        image_hash = hashlib.sha256(image_bytes or b"").hexdigest()
        record = {
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "predicted_species": species_name,
            "model_confidence": float(prediction["confidence"]),
            "top3": [item["class_name"] for item in prediction.get("top_k", [])[:3]],
            "preprocessing": prediction.get("preprocessing", {}),
            "verdict": verdict,
            "mismatches": mismatches,
            "observed_size": measured_size,
            "location": location,
            "alternative_identification": expert_species,
            "notes": notes,
            "image_sha256": image_hash,
            "image_retained": bool(allow_image and image_bytes),
        }
        with (FEEDBACK_DIR / "feedback.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        if allow_image and image_bytes:
            image_folder = FEEDBACK_DIR / "images"
            image_folder.mkdir(parents=True, exist_ok=True)
            (image_folder / f"{image_hash}.jpg").write_bytes(image_bytes)
        st.success("Feedback saved. It will be available for the next model review.")


def mini_result_card(title: str, prediction: Dict[str, Any]) -> None:
    confidence = float(prediction["confidence"])
    st.markdown(
        f"""
        <div class="mini-result">
            <div class="mini-title">{escape(title)}</div>
            <div class="mini-label">{escape(str(prediction["prediction"]))}</div>
            <div class="mini-confidence">Confidence: <strong>{format_percent(confidence)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_data_url(image_bytes: bytes) -> str:
    image = load_image_from_bytes(image_bytes)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def preview_image(image_bytes: bytes, caption: str, compact: bool = False) -> None:
    image = load_image_from_bytes(image_bytes)
    del image
    class_name = "image-frame compact" if compact else "image-frame"
    st.markdown(
        f"""
        <div class="{class_name}">
            <img src="{image_data_url(image_bytes)}" alt="{escape(caption)}">
            <div class="image-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def store_first_upload(uploaded_file) -> None:
    image_bytes = uploaded_file.getvalue()
    if image_bytes != st.session_state.first_image_bytes:
        for key in list(st.session_state.keys()):
            if key.startswith("second_uploader_"):
                del st.session_state[key]
        st.session_state.pair_nonce = int(st.session_state.get("pair_nonce", 0)) + 1
        st.session_state.first_image_bytes = image_bytes
        st.session_state.first_image_name = uploaded_file.name
        st.session_state.first_prediction = None
        st.session_state.second_image_bytes = None
        st.session_state.second_image_name = None
        st.session_state.second_prediction = None
        st.session_state.combined_prediction = None
        st.session_state.same_specimen_confirmed = False


def store_second_upload(uploaded_file) -> None:
    image_bytes = uploaded_file.getvalue()
    if image_bytes != st.session_state.second_image_bytes:
        st.session_state.second_image_bytes = image_bytes
        st.session_state.second_image_name = uploaded_file.name
        st.session_state.second_prediction = None
        st.session_state.combined_prediction = None


def show_error(message: str, exc: Exception) -> None:
    st.error(message)
    with st.expander("Technical details"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def predict_if_needed(state_key: str, image_key: str, artifacts) -> None:
    if st.session_state[state_key] is None and st.session_state[image_key] is not None:
        st.session_state[state_key] = predict_image_bytes(
            st.session_state[image_key],
            artifacts=artifacts,
            top_k=3,
        )


def show_model_information(artifacts) -> None:
    threshold = artifact_confidence_threshold(artifacts)
    joint_threshold = artifact_joint_confidence_threshold(artifacts)
    with st.expander("Model information"):
        st.markdown(
            "\n".join(
                [
                    f"- Model architecture: {artifacts.config.get('model_name', 'MobileNetV2')}",
                    f"- Number of classes: {len(artifacts.class_names)}",
                    f"- Device: {artifacts.device}",
                    f"- Artifact version: {artifacts.artifact_version}",
                    f"- Validation-balanced confidence threshold: {format_percent(threshold)}",
                    (
                        f"- Two-view threshold: {format_percent(joint_threshold)}"
                        if joint_threshold is not None
                        else "- Two-view threshold: not calibrated; fused scores are supporting evidence only"
                    ),
                    f"- Supported image formats: {', '.join(file_type.upper() for file_type in SUPPORTED_UPLOAD_TYPES)}",
                    "- Two-image method: same-individual joint evidence restricted to Image 1's Top-3 candidates.",
                ]
            )
        )


def navigate_to(page: str) -> None:
    st.session_state.page = page


def show_home_page() -> None:
    text_col, image_col = st.columns([1, 1.05], gap="large")
    with text_col:
        st.markdown('<div class="hero-kicker">Image-based species identification</div>', unsafe_allow_html=True)
        st.markdown('<h1 class="hero-title">Caddisfly<br><em>Image Classifier</em></h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-description">Upload a caddisfly photograph to get a predicted species and confidence score. Compare the result with reference images, or add a second view of the same specimen when the prediction is uncertain.</p>', unsafe_allow_html=True)
        st.button("Start identification", type="primary", on_click=navigate_to,
                  args=("identify",), use_container_width=False)
        st.caption("For curious observers, students and researchers.")
    with image_col:
        image_path = PACKAGE_ROOT / "web_assets/home_specimen.jpg"
        if image_path.is_file():
            st.markdown(
                f'<div class="specimen-plate"><img src="{"data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")}" '
                'alt="Caddisfly resting on a plant stem">'
                '<div class="plate-label"><span class="plate-species">Caddisfly photograph</span>'
                '<span>Project image</span></div></div>', unsafe_allow_html=True,
            )
    st.divider()
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown('<div class="editorial-note"><div class="section-kicker">Supported classes</div><h3>What can this website identify?</h3><p>The model predicts among 25 classification labels, including the mottled form of <i>Asmicridea edwardsii</i>. Results include species, family and reference photographs.</p></div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="editorial-note"><div class="section-kicker">Using the results</div><h3>A prediction to help you compare.</h3><p>Confidence expresses the model&#39;s preference among its supported classes. Check the reference images; unfamiliar species and uncertain results require expert review.</p></div>', unsafe_allow_html=True)


def show_first_image_workflow(artifacts) -> None:
    st.markdown('<div class="section-kicker">Specimen photograph</div>', unsafe_allow_html=True)
    st.title("Identify a caddisfly")
    st.write("Choose a clear photograph with the whole specimen visible.")
    with st.container(border=False):
        first_file = st.file_uploader(
            "Upload your specimen photograph", type=SUPPORTED_UPLOAD_TYPES,
            key=f"first_uploader_{st.session_state.uploader_nonce}",
            help="JPG, JPEG, PNG or WEBP. Keep the wings in focus and use a simple background.",
        )
        if first_file is not None:
            store_first_upload(first_file)
        if st.session_state.first_image_bytes is not None:
            preview_image(st.session_state.first_image_bytes,
                          st.session_state.first_image_name or "Your specimen")
        st.caption("Use even lighting, a simple background and a complete view of the specimen.")
        if st.button("Identify specimen", type="primary", use_container_width=True,
                     disabled=st.session_state.first_image_bytes is None):
            with st.spinner("Examining your specimen…"):
                predict_if_needed("first_prediction", "first_image_bytes", artifacts)
            navigate_to("result")
            st.rerun()
        if st.session_state.first_image_bytes is not None:
            st.button("Choose a different image", on_click=reset_application)


def show_result_page(artifacts) -> None:
    first = st.session_state.first_prediction
    if first is None:
        st.info("Upload a specimen photograph to see its prediction.")
        st.button("Upload a photograph", on_click=navigate_to, args=("identify",))
        return
    st.markdown('<div class="section-kicker">Prediction result</div>', unsafe_allow_html=True)
    st.title("Your specimen prediction")
    if st.session_state.second_prediction is not None and st.session_state.same_specimen_confirmed:
        show_joint_result_page(artifacts)
    else:
        threshold = artifact_confidence_threshold(artifacts)
        uncertain = requires_second_image(first["confidence"], threshold)
        image_col, result_col = st.columns([1.05, 1], gap="large")
        with image_col:
            preview_image(st.session_state.first_image_bytes,
                          st.session_state.first_image_name or "Your specimen")
        with result_col:
            result_card(first, threshold, note="Compare visible features with the reference specimens. This prediction is not an independently verified identification.")
            show_top3(first["top_k"], "Other candidates", expanded=False)
            if uncertain:
                st.warning("This view is not sufficient for a confident prediction. Add another angle of the same specimen.")
                st.button("Add another view", type="primary", on_click=navigate_to,
                          args=("supplement",), use_container_width=True)
        if not uncertain:
            show_species_profile(first, st.session_state.first_image_bytes)
    st.divider()
    new_col, home_col = st.columns(2)
    with new_col:
        st.button("Identify a new specimen", on_click=reset_application, use_container_width=True)
    with home_col:
        st.button("Return to home", on_click=navigate_to, args=("home",), use_container_width=True)


def show_second_image_workflow(artifacts) -> None:
    threshold = artifact_confidence_threshold(artifacts)
    first_prediction = st.session_state.first_prediction
    if first_prediction is None or not requires_second_image(
        first_prediction["confidence"], threshold
    ):
        return

    st.markdown('<div class="section-kicker">Second specimen photograph</div>', unsafe_allow_html=True)
    st.title("Add another view")
    st.button("Back to result", on_click=navigate_to, args=("result",))
    preview_image(st.session_state.first_image_bytes, "Original specimen", compact=True)
    st.markdown(
        """
        <div class="helper-text">
        The next upload is linked to Image 1 as a second view of the same individual.
        Use a different angle with clearer body features, better lighting, and less
        background obstruction. The final prediction will be selected only from
        Image 1's Top-3 candidates. Do not upload a different insect.
        </div>
        """,
        unsafe_allow_html=True,
    )

    same_specimen_confirmed = st.checkbox(
        "I confirm that Image 2 shows the same individual caddisfly as Image 1.",
        key="same_specimen_confirmed",
        help=(
            "The joint prediction assumes both images show the same specimen and "
            "therefore share the same species."
        ),
    )
    if not same_specimen_confirmed:
        st.info("Confirm the same-specimen relationship before uploading Image 2.")
        return

    second_file = st.file_uploader(
        "Upload Image 2 — another view of the same specimen",
        type=SUPPORTED_UPLOAD_TYPES,
        key=(
            f"second_uploader_{st.session_state.uploader_nonce}_"
            f"{st.session_state.pair_nonce}"
        ),
        help="Image 2 will be evaluated jointly with the retained Image 1 evidence.",
    )
    if second_file is not None:
        if second_file.getvalue() == st.session_state.first_image_bytes:
            st.warning("Image 2 is identical to Image 1. Upload a different view of the same specimen.")
            return
        store_second_upload(second_file)

    if st.session_state.second_image_bytes is None:
        st.info("Upload one different view of the same specimen. The app will not request additional images.")
        return

    preview_image(st.session_state.second_image_bytes,
                  st.session_state.second_image_name or "Second view")
    if st.button("Analyse both views", type="primary", use_container_width=True):
        with st.spinner("Comparing both views…"):
            predict_if_needed("second_prediction", "second_image_bytes", artifacts)
        navigate_to("result")
        st.rerun()


def show_joint_result_page(artifacts) -> None:
    threshold = artifact_confidence_threshold(artifacts)
    first_prediction = st.session_state.first_prediction
    second_prediction = st.session_state.second_prediction

    result_cols = st.columns(2, gap="large")
    with result_cols[0]:
        st.markdown("**Image 1 result**")
        preview_image(st.session_state.first_image_bytes, st.session_state.first_image_name or "First image", compact=True)
        mini_result_card("Image 1", first_prediction)
        show_top3(first_prediction["top_k"], "Image 1 Top-3 candidates", expanded=False)
    with result_cols[1]:
        st.markdown("**Image 2 result**")
        preview_image(st.session_state.second_image_bytes, st.session_state.second_image_name or "Second image", compact=True)
        st.caption("Image 2 supplies evidence for the retained Top-3; it does not introduce new candidates.")

    candidate_indices = [int(item["class_index"]) for item in first_prediction["top_k"]]
    second_candidate_prediction = enrich_species_summary(
        summarize_probabilities(
            restrict_probabilities_to_candidates(
                second_prediction["probabilities"], candidate_indices
            ),
            artifacts.class_names,
            k=3,
        ),
        artifacts,
    )
    with result_cols[1]:
        mini_result_card("Image 2 support within retained candidates", second_candidate_prediction)
        show_top3(
            second_candidate_prediction["top_k"],
            "Image 2 support for Image 1 Top-3",
            expanded=False,
        )
    combined_probabilities = combine_within_candidate_classes(
        first_prediction["probabilities"],
        st.session_state.second_prediction["probabilities"],
        candidate_indices,
    )
    st.session_state.combined_prediction = enrich_species_summary(
        summarize_probabilities(
            combined_probabilities,
            artifacts.class_names,
            k=3,
        ),
        artifacts,
    )

    joint_threshold = artifact_joint_confidence_threshold(artifacts)
    show_combined_result(
        first_prediction,
        second_candidate_prediction,
        st.session_state.combined_prediction,
        joint_threshold if joint_threshold is not None else threshold,
        threshold_is_calibrated=joint_threshold is not None,
    )
    show_species_profile(
        st.session_state.combined_prediction, st.session_state.first_image_bytes,
        first_prediction=first_prediction, second_prediction=second_prediction,
    )


def show_combined_result(
    first_prediction: Dict[str, Any],
    second_prediction: Dict[str, Any],
    combined_prediction: Dict[str, Any],
    threshold: float = CONFIDENCE_THRESHOLD,
    *,
    threshold_is_calibrated: bool = True,
) -> None:
    first_label = first_prediction["prediction"]
    second_label = second_prediction["prediction"]
    combined_confidence = float(combined_prediction["confidence"])
    accepted = threshold_is_calibrated and joint_result_is_accepted(
        first_label, second_label, combined_confidence, threshold
    )

    if first_label == second_label:
        agreement_note = "Both images agree."
    else:
        agreement_note = "The two images produced different predictions."

    if not threshold_is_calibrated:
        confidence_note = (
            "The fused score is supporting evidence only because a validated "
            "two-view acceptance threshold is not yet available."
        )
    elif combined_confidence < threshold:
        confidence_note = "The combined result is still uncertain. Please use a clearer image or consult an expert."
    elif first_label == second_label:
        confidence_note = "The shared class is supported by both views and is used as the final prediction."
    else:
        confidence_note = "The joint result passes the confidence threshold, but the image-level disagreement should be reviewed."

    st.markdown('<div class="compact-heading">Same-specimen joint result</div>', unsafe_allow_html=True)
    result_card(
        combined_prediction,
        threshold,
        heading="Final joint prediction",
        note=f"{agreement_note} {confidence_note}",
        accepted=accepted,
        status_override=(
            ("Joint score — review required", "status-warn")
            if not threshold_is_calibrated
            else None
        ),
    )

    if not threshold_is_calibrated:
        st.info(
            "Use the joint result as supporting evidence; it has not been calibrated for automatic acceptance."
        )
    elif combined_confidence < threshold:
        st.warning("The combined result is still uncertain. Please use a clearer image or consult an expert.")
    elif first_label != second_label:
        st.warning("The two images produced different predictions. Review the joint result carefully.")
    else:
        st.success("Both images agree.")

    show_top3(
        combined_prediction["top_k"],
        "Other joint candidates",
        expanded=False,
    )


def main() -> None:
    apply_page_style()
    initialise_session_state()
    with st.container(key="site_navigation"):
        brand_col, home_col, identify_col = st.columns([5, 1, 1.5])
        with brand_col:
            st.markdown('<div class="site-brand">Caddisfly<span>Image classifier</span></div>', unsafe_allow_html=True)
        with home_col:
            st.button("Home", on_click=navigate_to, args=("home",), use_container_width=True)
        with identify_col:
            st.button("Identify", on_click=navigate_to, args=("identify",), use_container_width=True)
    st.divider()

    if st.session_state.page == "home":
        show_home_page()
    else:
        try:
            artifacts = get_model_artifacts(active_model_signature())
        except Exception as exc:
            show_error("The model could not be loaded. Check that the packaged model files are present.", exc)
            return
        try:
            if st.session_state.page == "identify":
                show_first_image_workflow(artifacts)
            elif st.session_state.page == "supplement":
                show_second_image_workflow(artifacts)
            else:
                show_result_page(artifacts)
        except InferenceError as exc:
            show_error("Prediction could not be completed. Please choose a readable specimen image and try again.", exc)
        except Exception as exc:
            show_error("An unexpected application error occurred. Please try again.", exc)

    st.markdown('<div class="footer-note">Caddisfly Specimen Explorer · An educational and research project.<br>Model predictions should be checked against visible features and expert taxonomic assessment.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
