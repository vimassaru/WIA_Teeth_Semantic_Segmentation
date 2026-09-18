"""Gera `web/metrics_data.js` a partir dos CSVs de treino e do JSON de
métricas finais, pra página web não depender de servir a pasta `data/`
(o `python -m http.server` roda com raiz em `web/`, então arquivos fora
dali não são acessíveis via fetch).

Rodar (dentro do venv, na raiz do projeto):
    python src/tensorflow/build_metrics_js.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "data" / "results"
OUTPUT_PATH = ROOT / "web" / "metrics_data.js"


def read_curve(csv_path: Path) -> dict[str, list[float]]:
    epochs, test_loss, mean_iou = [], [], []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            test_loss.append(float(row["test_loss"]))
            mean_iou.append(float(row["mean_iou"]))
    return {"epoch": epochs, "test_loss": test_loss, "mean_iou": mean_iou}


def main() -> None:
    data = {
        "runs": {
            "sem_augmentacao": read_curve(RESULTS_DIR / "tensorflow_training_metrics_v1_no_augment.csv"),
            "com_augmentacao": read_curve(RESULTS_DIR / "tensorflow_training_metrics.csv"),
        },
        "final": json.loads((RESULTS_DIR / "tensorflow_final_metrics.json").read_text(encoding="utf-8")),
    }
    OUTPUT_PATH.write_text("window.METRICS_DATA = " + json.dumps(data) + ";\n", encoding="utf-8")
    print(f"Gerado {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
