"""Carrega o SegFormer pré-treinado via `keras_hub`.

No notebook em PyTorch, o checkpoint era carregado com a biblioteca
`transformers` da Hugging Face:

    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint, id2label=id2label, label2id=label2id,
    )

Na versão atual do `transformers` (5.x) o suporte a TensorFlow foi removido
por completo da biblioteca (só resta PyTorch) — por isso usamos aqui o
`keras_hub`, o pacote oficial da equipe Keras para modelos pré-treinados
(sucessor do keras-cv/keras-nlp). O preset `segformer_b0_ade20k_512` é o
SegFormer B0 (mesma família MiT-B0 do artigo original) já pré-treinado para
segmentação semântica no ADE20k — carregar com `num_classes=33` faz o
`keras_hub` trocar a última camada de classificação automaticamente,
mantendo o encoder pré-treinado (equivalente ao `ignore_mismatched_sizes` da
`transformers`).

Diferença de API que vale notar para quem vem do PyTorch: aqui não existe um
`nn.Module` — `SegFormerImageSegmenter` já é um `keras.Model` funcional
completo, com `.summary()`, `.fit()`, `.save()` de graça, e a saída já vem
redimensionada para a resolução cheia da imagem de entrada (o SegFormer em
PyTorch devolve logits em H/4 x W/4 e cabe a você fazer o upsample).
"""

from __future__ import annotations

import keras_hub

PRESET = "segformer_b0_ade20k_512"


def load_model(id2label: dict[int, str]) -> keras_hub.models.SegFormerImageSegmenter:
    return keras_hub.models.SegFormerImageSegmenter.from_preset(
        PRESET,
        num_classes=len(id2label),
    )


if __name__ == "__main__":
    from dataset import load_id2label

    model = load_model(load_id2label())
    model.summary()
