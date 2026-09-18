"""Calcula as métricas de avaliação usadas no artigo (seção II-D: Pixel
Accuracy, IoU, Dice Coefficient) para o modelo TensorFlow treinado, e salva
um JSON que a página web usa para montar os cards de métricas.

Rodar (dentro do venv em WSL, na raiz do projeto):
    python src/tensorflow/eval_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import keras
import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from dataset import build_dataset, load_id2label  # noqa: E402

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "data" / "tensorflow_checkpoint" / "segformer_teeth.keras"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "data" / "results" / "tensorflow_final_metrics.json"


def main() -> None:
    id2label = load_id2label()
    num_labels = len(id2label)

    model = keras.models.load_model(CHECKPOINT_PATH)
    test_ds = build_dataset("test", batch_size=2, shuffle=False)

    confusion = tf.zeros((num_labels, num_labels), dtype=tf.int64)
    for images, masks in test_ds:
        logits = model(images, training=False)
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        confusion += tf.math.confusion_matrix(
            tf.reshape(masks, [-1]),
            tf.reshape(predictions, [-1]),
            num_classes=num_labels,
            dtype=tf.int64,
        )

    cm = confusion.numpy()
    intersection = cm.diagonal()
    pred_total = cm.sum(axis=0)
    true_total = cm.sum(axis=1)
    union = true_total + pred_total - intersection

    # Pixel Accuracy: eq. (nenhum número no artigo, é so acertos/total).
    pixel_accuracy = float(intersection.sum() / cm.sum())

    # IoU por classe (eq. 2 do artigo) e a media, ignorando classes ausentes
    # no conjunto de teste (uniao = 0).
    with_union = union > 0
    per_class_iou = intersection[with_union] / union[with_union]
    mean_iou = float(per_class_iou.mean())

    # Dice / F1 (eq. 3 do artigo): 2*intersecao / (total de pixels das duas mascaras).
    dice_denominator = true_total + pred_total
    with_dice = dice_denominator > 0
    per_class_dice = (2 * intersection[with_dice]) / dice_denominator[with_dice]
    mean_dice = float(per_class_dice.mean())

    result = {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": mean_iou,
        "mean_dice": mean_dice,
        "num_classes": num_labels,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Salvo em {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
