from pathlib import Path
import joblib


def save_model(pipeline, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, str(p))


def load_model(path: str):
    return joblib.load(path)


__all__ = ["save_model", "load_model"]
