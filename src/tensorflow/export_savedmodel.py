"""Empacota o modelo treinado num `tf.Module` de inferência e salva como
SavedModel — o formato que o `tensorflowjs_converter` sabe ler.

A ideia é colocar dentro do próprio grafo tudo que a página HTML não deveria
ter que reimplementar em JavaScript: resize, normalização e o argmax final.
Assim o `web/index.html` só faz:

    tensor de pixels da imagem -> model.predict -> tensor de classes por pixel

sem duplicar a lógica de pré/pós-processamento em duas linguagens.

Rodar (dentro do venv em WSL, na raiz do projeto):
    python src/tensorflow/export_savedmodel.py
"""

from __future__ import annotations

from pathlib import Path

import keras
import tensorflow as tf

from dataset import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

CHECKPOINT_PATH = Path(__file__).resolve().parents[2] / "data" / "tensorflow_checkpoint" / "segformer_teeth.keras"
SAVEDMODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "tensorflow_savedmodel"


class InferenceModel(tf.Module):
    """Equivalente a um `nn.Module` de inferência no PyTorch, mas aqui a
    "forward" vira um método decorado com `@tf.function` e uma
    `input_signature` explícita — é essa assinatura que define o formato de
    entrada que o modelo vai aceitar no navegador via TensorFlow.js."""

    def __init__(self, segformer: keras.Model):
        super().__init__()
        self.segformer = segformer

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8)]
    )
    def predict_mask(self, image_uint8: tf.Tensor) -> tf.Tensor:
        original_hw = tf.shape(image_uint8)[1:3]

        image = tf.image.resize(tf.cast(image_uint8, tf.float32), IMAGE_SIZE, method="bilinear")
        image = image / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD  # NHWC, igual ao treino

        logits = self.segformer(image, training=False)  # (N, 512, 512, num_classes)
        logits = tf.image.resize(logits, original_hw, method="bilinear")

        return tf.argmax(logits, axis=-1, output_type=tf.int32)  # (N, H, W)


def main() -> None:
    segformer = keras.models.load_model(CHECKPOINT_PATH)
    inference_model = InferenceModel(segformer)

    SAVEDMODEL_DIR.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        inference_model,
        str(SAVEDMODEL_DIR),
        signatures={"serving_default": inference_model.predict_mask},
    )
    print(f"SavedModel salvo em {SAVEDMODEL_DIR}")
    print(
        "Próximo passo (dentro do venv):\n"
        f"  tensorflowjs_converter --input_format=tf_saved_model "
        f"{SAVEDMODEL_DIR} web/model"
    )


if __name__ == "__main__":
    main()
