import os
import json
from typing import Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import gradio as gr

from config import NUM_LANDMARKS, ANATOMICAL_LANDMARKS


IMAGE_SIZE = 256
MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LandmarkModel(nn.Module):
    def __init__(self, num_landmarks: int):
        super().__init__()
        # We load trained weights from MODEL_FILENAME, so avoid downloading pretrained
        # weights at runtime (more reliable in Spaces).
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=False)

        backbone.fc = nn.Linear(512, num_landmarks * 2)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


_model: LandmarkModel | None = None


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    # Common patterns:
    # - raw state_dict (dict of tensors)
    # - {'state_dict': state_dict} or {'model_state_dict': state_dict}
    # - full model object (rare)
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj.keys()) and any(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            return obj  # already a state_dict

        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj:
                inner = obj[key]
                if hasattr(inner, "state_dict"):
                    return inner.state_dict()  # type: ignore[no-any-return]
                if isinstance(inner, dict):
                    return inner  # type: ignore[no-any-return]

    if hasattr(obj, "state_dict"):
        return obj.state_dict()  # type: ignore[no-any-return]

    raise TypeError(
        "Unsupported checkpoint format. Expected a PyTorch state_dict or a dict containing one."
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle DataParallel checkpoints where keys are like 'module.backbone.conv1.weight'
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _resolve_weights_path() -> str:
    """Return a local path to model weights.

    Priority:
    1) Local file named MODEL_FILENAME (default: model.pth)
    2) Download from HF Hub if HF_MODEL_REPO is set

    Env vars (optional):
    - HF_MODEL_REPO: e.g. "username/my-ceph-model" (a HF *model* repo)
    - HF_MODEL_FILENAME: defaults to MODEL_FILENAME
    - HF_MODEL_REVISION: optional branch/tag/commit
    - HF_TOKEN / HUGGINGFACE_HUB_TOKEN: if the model repo is private
    """

    if os.path.exists(MODEL_FILENAME):
        return MODEL_FILENAME

    repo_id = os.environ.get("HF_MODEL_REPO")
    if not repo_id:
        return MODEL_FILENAME

    filename = os.environ.get("HF_MODEL_FILENAME", MODEL_FILENAME)
    revision = os.environ.get("HF_MODEL_REVISION")

    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required to download weights from the Hub. "
            "Add it to requirements.txt or upload model.pth into the Space repo."
        ) from e

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        revision=revision,
    )


def _load_model() -> LandmarkModel:
    global _model
    if _model is not None:
        return _model

    model = LandmarkModel(NUM_LANDMARKS)

    weights_path = _resolve_weights_path()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Missing model weights. Tried '{MODEL_FILENAME}' (local) and also checked "
            "HF_MODEL_REPO (if set). Upload weights to the Space, or set HF_MODEL_REPO "
            "to download them from the Hugging Face Hub."
        )

    raw = torch.load(weights_path, map_location="cpu")
    state = _strip_module_prefix(_extract_state_dict(raw))
    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()

    _model = model
    return _model


def _preprocess(pil_img: Image.Image) -> torch.Tensor:
    img = pil_img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,3,H,W)
    return tensor


def _symbols_in_order() -> list[str]:
    # Python 3.7+ preserves dict insertion order.
    return [v["symbol"] for v in ANATOMICAL_LANDMARKS.values()]


def _calculate_angle(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> float:
    a = np.array(A, dtype=np.float32)
    b = np.array(B, dtype=np.float32)
    c = np.array(C, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return float("nan")

    cos_angle = float(np.dot(ba, bc) / denom)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _diagnosis_from_anb(anb: float) -> str:
    # Common clinical heuristic:
    # - ANB > 4  => Class II
    # - ANB < 0  => Class III
    # - else     => Class I
    if np.isnan(anb):
        return "Unknown (invalid angle)"
    if anb > 4:
        return "Class II (Retruded mandible)"
    if anb < 0:
        return "Class III (Prognathic mandible)"
    return "Class I (Normal)"


def predict(pil_img: Image.Image) -> Dict[str, Any]:
    model = _load_model()

    x = _preprocess(pil_img).to(DEVICE)

    with torch.no_grad():
        preds = model(x).detach().cpu().numpy().reshape(NUM_LANDMARKS, 2)

    # Model was trained on normalized coords (0..1) in 256x256 space.
    preds = np.clip(preds, 0.0, 1.0)
    coords_256 = preds * float(IMAGE_SIZE)

    symbols = _symbols_in_order()
    lm_dict: Dict[str, Tuple[float, float]] = {
        symbols[i]: (float(coords_256[i, 0]), float(coords_256[i, 1])) for i in range(NUM_LANDMARKS)
    }

    # Angles
    S = lm_dict.get("S")
    N = lm_dict.get("N")
    A = lm_dict.get("A")
    B = lm_dict.get("B")

    if not (S and N and A and B):
        raise KeyError("Required landmarks S, N, A, B not found in landmark mapping.")

    sna = _calculate_angle(S, N, A)
    snb = _calculate_angle(S, N, B)
    anb = sna - snb

    diagnosis = _diagnosis_from_anb(anb)

    return {
        "SNA": sna,
        "SNB": snb,
        "ANB": anb,
        "Diagnosis": diagnosis,
        "Landmarks_256px": lm_dict,
    }


def predict_ui(pil_img: Image.Image):
    try:
        out = predict(pil_img)
        report = "\n".join(
            [
                f"SNA: {out['SNA']:.2f}",
                f"SNB: {out['SNB']:.2f}",
                f"ANB: {out['ANB']:.2f}",
                f"Diagnosis: {out['Diagnosis']}",
            ]
        )
        return report, out
    except Exception as e:
        # Make errors visible in Spaces logs + UI.
        return f"ERROR: {type(e).__name__}: {e}", {"error": str(e)}


demo = gr.Interface(
    fn=predict_ui,
    inputs=gr.Image(type="pil", label="Upload X-ray"),
    outputs=[
        gr.Textbox(label="Report"),
        gr.JSON(label="Raw output"),
    ],
    title="Cephalometric Landmark → Angle Prediction",
    description=(
        "Uploads a lateral cephalogram, predicts landmarks, then computes SNA/SNB/ANB and skeletal class. "
        "API is enabled (see the Space 'Use via API' section)."
    ),
    allow_flagging="never",
    api_name="predict",
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))
