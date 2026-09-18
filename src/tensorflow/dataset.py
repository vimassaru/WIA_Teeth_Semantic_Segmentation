"""Carrega o dataset de raio-x panorâmico como tf.data.Dataset.

Equivalente TensorFlow do que o notebook em PyTorch fazia com
`datasets.Dataset` + `SegformerImageProcessor`. Principais diferenças de API
para quem vem do PyTorch:

- `torch.utils.data.Dataset.__getitem__` vira uma função Python normal que a
  gente "encaixa" no grafo com `tf.data.Dataset.map(...)`. O TF chama essa
  função construindo um grafo (tf.function) por trás dos panos, então
  operações têm que ser do tipo `tf.image.*` / `tf.io.*` em vez de PIL puro
  quando possível, senão é preciso `tf.py_function` (mais lento).
- `DataLoader(dataset, batch_size=...)` vira `dataset.batch(...)`. Não existe
  um objeto "DataLoader" separado — o próprio `tf.data.Dataset` já expõe
  `.batch()`, `.shuffle()`, `.prefetch()` encadeáveis.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tensorflow as tf

DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "tooth_label_mask_segmentation"

# Mesmos valores usados pelo `SegformerImageProcessor` (ImageNet mean/std),
# porque carregamos o checkpoint pré-treinado `nvidia/mit-b0` e ele espera
# entradas normalizadas do mesmo jeito que no pré-treino original.
IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)

# O SegFormer original foi treinado com entradas quadradas; usamos 512x512
# como no notebook em PyTorch (do_resize=True do SegformerImageProcessor).
IMAGE_SIZE = (512, 512)


def load_id2label(split: str = "train") -> dict[int, str]:
    """Lê `_classes.csv` (mesmo arquivo gerado pelo Roboflow usado no PyTorch)."""
    csv_path = DATA_ROOT / split / "_classes.csv"
    id2label: dict[int, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # cabeçalho "Pixel Value, Class"
        for row in reader:
            pixel_value, label = (cell.strip() for cell in row)
            id2label[int(pixel_value)] = label
    return id2label


def _list_pairs(split: str) -> tuple[list[str], list[str]]:
    image_dir = DATA_ROOT / split / "image"
    mask_dir = DATA_ROOT / split / "mask"
    image_paths, mask_paths = [], []
    for image_path in sorted(image_dir.glob("*.jpg")):
        mask_path = mask_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            raise FileNotFoundError(f"Máscara ausente para {image_path.name}: {mask_path}")
        image_paths.append(str(image_path))
        mask_paths.append(str(mask_path))
    return image_paths, mask_paths


def _decode_pair(image_path: tf.Tensor, mask_path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    # keras_hub segue o padrão do Keras: canais por último (NHWC), diferente
    # do NCHW que a `transformers` (PyTorch) usa por padrão.

    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    # nearest-neighbor no resize da máscara: interpolar valores de classe
    # (como faria um bilinear) criaria classes inexistentes entre pixels.
    mask = tf.image.resize(mask, IMAGE_SIZE, method="nearest")
    mask = tf.cast(tf.squeeze(mask, axis=-1), tf.int32)

    return image, mask


def _normalize(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return image, mask


# Espelhar a imagem (flip horizontal) troca o lado da boca na foto: o que
# era o quadrante direito do paciente passa a aparecer como esquerdo. Como
# cada classe de dente já codifica o quadrante (11-18 = Q1, 21-28 = Q2,
# 31-38 = Q3, 41-48 = Q4 -> ids 1-8, 9-16, 17-24, 25-32), o flip sem trocar
# os rótulos ensinaria o modelo errado (um "dente 11" viraria pixel de
# "dente 21" na imagem espelhada). Esta tabela remapeia Q1<->Q2 e Q3<->Q4,
# mantendo a posição (1º incisivo continua 1º incisivo etc.) e o fundo (0)
# fixo.
_FLIP_REMAP = tf.constant(
    [0] + list(range(9, 17)) + list(range(1, 9)) + list(range(25, 33)) + list(range(17, 25)),
    dtype=tf.int32,
)


def _augment(image: tf.Tensor, mask: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Augmentação só pro split de treino. Chamada antes da normalização
    ImageNet, sobre a imagem ainda em [0, 1] (mais fácil de calibrar
    brilho/contraste nessa escala)."""
    if tf.random.uniform([]) < 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask[..., tf.newaxis])[..., 0]
        mask = tf.gather(_FLIP_REMAP, mask)

    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, mask


def build_dataset(split: str, batch_size: int, shuffle: bool, augment: bool = False) -> tf.data.Dataset:
    image_paths, mask_paths = _list_pairs(split)
    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)
    ds = ds.map(_decode_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_normalize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


if __name__ == "__main__":
    id2label = load_id2label()
    print(f"{len(id2label)} classes:", id2label)
    train_ds = build_dataset("train", batch_size=4, shuffle=True)
    images, masks = next(iter(train_ds))
    print("images", images.shape, images.dtype)  # (N, 512, 512, 3) NHWC
    print("masks", masks.shape, masks.dtype, "valores únicos:", np.unique(masks.numpy()))
