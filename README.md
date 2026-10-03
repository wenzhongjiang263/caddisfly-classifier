# Caddisfly Image Classifier

A graduation project developed in collaboration with a museum to support photograph-based caddisfly identification.

**[Open the website](https://caddisfly-classifier.streamlit.app/)**

Visitors can use the website directly in a desktop or mobile browser. No software installation or model download is required on the visitor's device.

## Features

- Upload a specimen photograph and view its predicted species, family and confidence score.
- Compare the prediction with labelled reference photographs.
- Add another view of the same specimen when the first prediction is uncertain.
- Explore recorded distribution and observation months.
- Enable map interaction explicitly; maps otherwise allow normal page scrolling.

## Model and interpretation

The application uses a ConvNeXt-Tiny multi-task classifier with 512-pixel inputs and 25 classification labels, including the mottled form of *Asmicridea edwardsii*. Two YOLO detectors support conservative specimen cropping. Inference includes horizontal-flip test-time augmentation and validation-calibrated single-image confidence thresholds.

The 25 labels do not represent 25 distinct biological species. Confidence expresses the model's preference among supported labels; it does not independently verify identification or reliably reject unfamiliar species. Two-view scores are supporting evidence because an independent two-view acceptance threshold has not been calibrated.

## Repository contents

| Path | Purpose |
| --- | --- |
| `cloud_app.py` | Streamlit Community Cloud entry point |
| `app.py` | Website interface and session workflow |
| `inference.py` | Model loading, preprocessing and predictions |
| `model_assets.py` | Verified model downloads |
| `caddisfly_v2/` | Shared model, label and transform code |
| `model/` | Active model selector and deployment metadata |
| `deployment/` | Model checksums, configuration example and initial verification record |
| `web_assets/` | Homepage image, species profiles and reference photographs |
| `requirements.txt` | CPU runtime dependencies |
| `packages.txt` | Linux system dependencies |
| `.streamlit/config.toml` | Theme and server settings |

This repository contains the website runtime. Original datasets, training caches, historical checkpoints and machine-specific virtual environments are excluded.

## Model files

Download the three assets from [Model Release v1](https://github.com/wenzhongjiang263/caddisfly-classifier/releases/tag/model-v1) when running locally:

- `last_deploy_calibrated.pt`
- `detector_yolo11n_512.pt`
- `detector_yolo11n_1024.pt`

Place them in `model/convnext_tiny_detector512/` without changing their names. On Community Cloud, the application downloads them automatically before the first identification. File sizes and SHA-256 checksums are checked against `deployment/model_assets.json`; incomplete or invalid downloads do not replace existing files.

## Deploy on Streamlit Community Cloud

1. Sign in at [Streamlit Community Cloud](https://share.streamlit.io/) and connect this GitHub repository.
2. Select branch `main` and entry point `cloud_app.py`.
3. In Advanced settings, select Python 3.12 for the locally tested runtime and add the configuration below to Secrets.
4. Deploy and verify the complete identification workflow.

```toml
[deployment]
release_base_url = "https://github.com/wenzhongjiang263/caddisfly-classifier/releases/download/model-v1"
```

The current hosted application was deployed with Python 3.14.7, and its identification workflow and mobile access were confirmed by the project owner. Python 3.12 remains the locally tested version.

## Local use

Create a Python 3.12 virtual environment and install the dependencies:

```bash
python -m venv .venv
# Activate the environment using the command appropriate for your operating system.
python -m pip install -r requirements.txt
streamlit run app.py
```

For local use, download the model files as described above. To reproduce cloud mode, save the deployment configuration in `.streamlit/secrets.toml` and run `streamlit run cloud_app.py`. Do not commit that local configuration file.

## Performance and limitations

- Cloud inference uses CPU. Response time depends on image size, available resources and concurrent requests.
- Display previews are resized JPEGs cached within each user's session. Original uploaded bytes are retained for inference.
- Uploads are limited to 25 MB in the cloud configuration.
- The public demo hides feedback submission because local cloud storage is not used for long-term feedback retention.
- Reference images are labelled examples from the project's development data, not independent identification evidence.
- Free hosting may require a cold start. Open the site and perform a trial identification before a scheduled demonstration.

## Validation

The local project passed 45 tests after the preview optimization. Checks covered model loading, prediction outputs, two-view state, download verification and preservation of original inference inputs. The exported runtime was separately exercised with a real specimen photograph on CPU. Initial export checks are recorded in `deployment/verification.json`; that file describes the initial preparation stage rather than the current hosting status.

After deploying updates, check image upload, single-image results, uncertain-result handling, second-view analysis, reference images, map interaction and resetting for a new specimen. The project owner confirmed that the optimized public website was usable on mobile.

## Deployment references

- [Deploy an application](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Configure dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
