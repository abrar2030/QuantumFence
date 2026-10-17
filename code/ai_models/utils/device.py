from typing import Optional


def resolve_device(preference: str = "auto") -> str:
    pref = (preference or "auto").lower()
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        return "cpu"
    return "cpu"


def supports_half(device: Optional[str]) -> bool:
    return bool(device) and str(device).startswith("cuda")
