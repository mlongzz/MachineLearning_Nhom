"""JSON output and guards against mixing incompatible experiment runs."""
from pathlib import Path
import hashlib
import json
from datetime import timezone, timedelta
from .config import BASE, GRIDS, ROUNDS, NUM, FE, SEED

LOCAL_TIME = timezone(timedelta(hours=7))


def dump(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def configuration_signature():
    payload = json.dumps({"base": BASE, "grids": GRIDS, "rounds": ROUNDS,
                          "numeric": NUM, "engineered": FE, "seed": SEED},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def check_output(output, data):
    output = Path(output)
    signature = configuration_signature()
    digest = hashlib.sha256(Path(data).read_bytes()).hexdigest()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output is not empty. Choose a new directory to preserve previous runs.")
    output.mkdir(parents=True, exist_ok=True)
    return digest, signature
