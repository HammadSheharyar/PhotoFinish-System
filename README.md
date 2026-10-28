# PhotoFinish-System
# PhotoFinish System 🏁

Automated photo-finish winner detection and vertical-line estimation system. Detects athletes in finish-line camera frames, draws per-athlete vertical finish lines from a single learned keypoint, and reads race bib numbers — served as a production-ready API.

## Overview

Photo-finish races are decided by centimeters. This system automates the classical photo-finish workflow:

1. **Detect** all athletes (and their bibs) in a finish-line frame — RTMDet-M, fine-tuned for the photo-finish domain
2. **Estimate the crossing point** per athlete — RTMPose-Small with a single keypoint
3. **Draw vertical lines** through each athlete's crossing point, with confidence scores
4. **Read race bib numbers** (EasyOCR / PaddleOCR) and match each bib to its athlete
5. **Serve results** via a FastAPI endpoint, deployed as a Windows service

## Results

| Model | Metric | Value |
| --- | --- | --- |
| RTMDet-M (person + bib) | Test mAP@[.5:.95] | **79.7%** |
| RTMDet-M (person + bib) | Test mAP@50 | 99.2% |
| RTMDet-M (person + bib) | Person AP | 92.7% |
| RTMDet-M (person + bib) | Bib AP | 66.7% |
| RTMPose-Small (1 keypoint) | Test AP | 47.03% |
| RTMPose-Small (1 keypoint) | Test AR | 60.19% |

Detection evaluation includes full COCO diagnostics: confusion matrix, per-class precision/recall, and top-50 false-positive/false-negative image galleries.

## Architecture

```javascript
POST /predict (image)
        │
        ▼
┌───────────────────┐     ┌──────────────────┐
│   RTMDet-M        │────►│  person boxes    │
│  person + bib     │     │  (label 1)       │
│  detection        │     └────────┬─────────┘
└───────────────────┘              │
        │                          ▼
        │                   ┌──────────────────┐
        │                   │  RTMPose-Small   │
        │                   │  1 keypoint      │
        │                   │  per person      │
        │                   └────────┬─────────┘
        │                          │
        ▼                          ▼
┌──────────────────────────────────────────┐
│  vertical line at keypoint.x per athlete │
│  bib crop → OCR (EasyOCR / PaddleOCR)    │
│  bib-to-athlete spatial matching         │
└──────────────────────────────────────────┘
        │
        ▼
JSON: personId, boxes, lineCoordinates,
      confidences, bibNo, processingTimeMs
```

## Tech Stack

**Python · MMDetection · MMPose · RTMDet · RTMPose · PyTorch · CUDA · ONNX Runtime · EasyOCR / PaddleOCR · FastAPI · NSSM · OpenCV**

## API

```powershell
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

**`POST /predict`** — multipart form with an image file (optional `frameNumber`):

```json
{
  "frameNumber": 0,
  "imageWidth": 1229,
  "imageHeight": 2048,
  "persons": [
    {
      "personId": 1,
      "personCoordinates": {"x0": 531.7, "y0": 160.5, "x1": 674.3, "y1": 550.8},
      "lineCoordinates":  {"x0": 634.5, "y0": 0, "x1": 634.5, "y1": 2047},
      "personConfidence": 0.97,
      "lineConfidence":   0.86,
      "bibNo": "25"
    }
  ],
  "processingTimeMs": 1435.5
}
```

Other endpoints: `GET /health` (liveness probe), `GET /` (service banner). Runtime-tunable settings (thresholds, annotated-image saving) are exposed without restarts.

## Key Engineering Details

- **GPU safety**: all inference runs under a single `threading.Lock` — one shared GPU model set, strictly serialized requests, async I/O stays parallel.
- **Blackwell support (RTX 5050, sm_120)**: MMCV compiled from source against PyTorch 2.7.1+cu128 — prebuilt wheels don't ship sm_120 kernels.
- **Dual backends**: every endpoint works in both PyTorch and ONNX Runtime modes (`PHOTOFINISH_BACKEND` env var) for deployment flexibility.
- **Windows service deployment**: runs under NSSM with file-based logs, environment-driven configuration, and health-check integration.
- **Resumable training**: training runs resume from checkpoints; fragmented logs from interrupted runs are merged into continuous curves (fixing MMEngine's `epoch=0` validation-log quirk and duplicate resumed epochs) for clean TensorBoard monitoring.
- **Bib OCR**: crops are upscaled and enhanced with a CLAHE + Otsu preprocessing ladder, validated (digit-only, leading-zero stripping, confidence gating), and matched to athletes by spatial containment.

## Repository Layout

```javascript
├── app/                        # FastAPI service + inference engine
├── configs/                    # RTMDet / RTMPose training & inference configs
├── models/                     # trained checkpoints (detector + pose)
├── tests/test/                 # sample finish-line frames
└── service/                    # NSSM install/uninstall scripts
```

## Training

Detector and pose models are fine-tuned with MMDetection / MMPose:

```powershell
python tools\train.py configs\rtmdet\rtmdet_m_photofinish_person_bib_hr.py --amp --resume
python tools\test.py  configs\rtmdet\rtmdet_m_photofinish_person_bib_hr.py <best_ckpt.pth>
```

Evaluation includes COCO mAP, classwise AP, confusion matrices, and FP/FN image galleries via MMDetection analysis tools.

## Status

- ✅ Athlete detection + vertical-line drawing (production)
- ✅ FastAPI service + Windows service deployment
- 🔄 Bib-number OCR accuracy improvements in progress
