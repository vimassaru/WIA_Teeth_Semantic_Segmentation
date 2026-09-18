"""Fine-tuning do SegFormer em TensorFlow, espelhando o notebook em PyTorch.

Mapa mental PyTorch -> TensorFlow usado neste arquivo:

  PyTorch                                  TensorFlow (aqui)
  ----------------------------------       ----------------------------------
  loss.backward()                          tape.gradient(loss, variaveis)
  optimizer.step()                         optimizer.apply_gradients(...)
  optimizer.zero_grad()                    (não precisa: gradientes não se
                                             acumulam entre chamadas do
                                             GradientTape, cada `with` é novo)
  model.train() / model.eval()             não existe alternância global;
                                             camadas como Dropout recebem um
                                             argumento `training=True/False`
                                             explícito na chamada
  Trainer(...).train() do HF               poderíamos ter usado
                                             `model.compile()` + `model.fit()`
                                             (mais direto, e é o caminho
                                             idiomático em Keras), mas o loop
                                             manual abaixo deixa explícito o
                                             paralelo com o `Trainer` por trás
                                             dos panos e facilita logar o IoU
                                             por época do jeito que o artigo
                                             mostra (Fig. 14-19).

Diferente da versão baseada em `transformers`, o `keras_hub` já devolve os
logits na resolução cheia da imagem (512x512xNUM_CLASSES) — o SegFormer em
si prevê em H/4 x W/4 e o upsample já vem embutido no modelo, então não
precisamos fazer isso manualmente aqui.

Rodar (dentro do venv em WSL, na raiz do projeto):
    python src/tensorflow/train.py --epochs 200
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import tensorflow as tf

# No WSL a GPU é compartilhada com o compositor do Windows, então nem toda a
# VRAM da placa está livre. Por padrão o TF reserva quase toda a memória
# disponível de uma vez (o que aqui causava OOM por fragmentação); com
# `memory growth` ele aloca sob demanda, igual ao comportamento padrão do
# PyTorch (que já é "lazy" por natureza). Precisa rodar antes de qualquer
# outro import que toque a GPU (mesmo criar um `tf.constant` já inicializa o
# dispositivo), por isso vem antes dos imports do projeto.
for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from dataset import build_dataset, load_id2label  # noqa: E402
from model import load_model  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "data" / "tensorflow_checkpoint"


def mean_iou(cm: tf.Tensor) -> float:
    """Calcula o IoU médio a partir de uma matriz de confusão acumulada,
    equivalente ao `evaluate.load("mean_iou")` usado no notebook em PyTorch."""
    cm = cm.numpy()
    intersection = cm.diagonal()
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    with_union = union > 0
    return float((intersection[with_union] / union[with_union]).mean())


def run(epochs: int, batch_size: int, learning_rate: float) -> None:
    id2label = load_id2label()
    num_labels = len(id2label)

    train_ds = build_dataset("train", batch_size=batch_size, shuffle=True, augment=True)
    test_ds = build_dataset("test", batch_size=batch_size, shuffle=False)

    model = load_model(id2label)

    # Taxa de aprendizado decrescente (cosine decay) em vez de fixa: no
    # início dá passos maiores (convergência rápida) e vai encolhendo
    # suavemente até `alpha * learning_rate` na última época — ajuda a
    # refinar perto da convergência em vez de "pular" em torno do mínimo com
    # uma taxa alta constante, que era um padrão visível nos logs anteriores
    # (mean_iou oscilando entre 0.71-0.75 sem tendência de subir mais).
    steps_per_epoch = int(tf.data.experimental.cardinality(train_ds).numpy())
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.02,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS_DIR / "tensorflow_training_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "test_loss", "mean_iou", "seconds"])

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss_avg = tf.keras.metrics.Mean()
        for images, masks in train_ds:
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss = loss_fn(masks, logits)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            train_loss_avg.update_state(loss)

        test_loss_avg = tf.keras.metrics.Mean()
        confusion = tf.zeros((num_labels, num_labels), dtype=tf.int64)
        for images, masks in test_ds:
            logits = model(images, training=False)
            test_loss_avg.update_state(loss_fn(masks, logits))
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            confusion += tf.math.confusion_matrix(
                tf.reshape(masks, [-1]),
                tf.reshape(predictions, [-1]),
                num_classes=num_labels,
                dtype=tf.int64,
            )

        iou = mean_iou(confusion)
        elapsed = time.time() - start
        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss_avg.result():.4f} "
            f"test_loss={test_loss_avg.result():.4f} "
            f"mean_iou={iou:.4f} "
            f"({elapsed:.1f}s)"
        )
        with metrics_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, float(train_loss_avg.result()), float(test_loss_avg.result()), iou, elapsed]
            )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(CHECKPOINT_DIR / "segformer_teeth.keras")
    print(f"Modelo salvo em {CHECKPOINT_DIR / 'segformer_teeth.keras'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    # O artigo original usou batch_size=8 num Colab com GPU dedicada; aqui a
    # RTX 3070 é compartilhada com o desktop do Windows via WSL (~5.5GB
    # livres de 8GB), e batch_size=4 ainda estoura memória em algum ponto do
    # treino. batch_size=2 foi o maior valor estável nos testes.
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=6e-4)
    args = parser.parse_args()
    run(args.epochs, args.batch_size, args.learning_rate)
