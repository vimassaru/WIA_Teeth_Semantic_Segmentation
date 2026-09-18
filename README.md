# WIA
Workshop de Inteligência Artificial - UNIFESP

🔗 **Demo ao vivo**: [vimassaru.github.io/WIA_Teeth_Semantic_Segmentation](https://vimassaru.github.io/WIA_Teeth_Semantic_Segmentation/) — sobe uma panorâmica e vê a segmentação rodando 100% no navegador, sem servidor.

# Informações do Projeto

Este projeto foi desenvolvido na unidade curricular Inteligência Artificial da Unifesp, onde apliquei conceitos de machine learning para tarefa de segmentação semântica como as redes neurais artificiais e processamento de imagens.

O projeto **SmileDataAI**, é um modelo de aprendizado de máquina para tarefa de segmentação semântica em imagens panorâmicas de raio-x odontológicos. O modelo recebe uma *imagem* como input e retorna em seu output uma predição em formato de *máscara*. Segue abaixo uma predição realizada pelo modelo:

<img align="center" src="https://github.com/vimassaru/WIA/blob/main/data/images/pred_ground_truth.png">

*Artigo do projeto*: [SmileDataAI: segmentação semântica de dentes em imagens de raio-x por meio de aprendizado profundo.](SmileDataAI_segmentacao_semantica_de_dentes_em_imagens_por_meio_de_aprendizado_profundo.pdf)

# Tecnologias Utilizadas

Para o desenvolvimento de todo o projeto, foram utilizadas as seguintes tecnologias:
<div style="display: inline_block">
  <img align="center" alt="Icon-Python" height="100" width="200" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg">
  <img align="center" alt="Icon-Jupyter" height="100" width="200" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" />
  <img align="center" alt="Icon-PyTorch" height="100" width="200" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg"/>
  <img align="center" alt="Icon-TensorFlow" height="100" width="200" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg"/>
  <img align="center" alt="Icon-JavaScript" height="100" width="200" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg"/>
</div>


# Pipeline do Projeto

Abaixo está demonstrado o pipeline utilizado para o desenvolvimento da rede neural, destacando as etapas de treino e teste em <span style="color:yellow">amarelo</span> e <span style="color:yellow">amarelo</span>, respectivamente.

<img align="center" src="https://github.com/vimassaru/WIA/blob/main/data/images/SmileDataAI.png">

## Hugging Face - Modelo e Dataset

O meu modelo e o dataset tratado se encontram na página do Hugging Face nos seguintes links:

- [Dataset](https://huggingface.co/datasets/vimassaru/teethsegmentation/tree/main)
- [Arquitetura SegFormer](https://huggingface.co/vimassaru/segformer-b0-finetuned-segments-sidewalk-oct-22)
- [Arquitetura UPerNet Convnext Tiny](https://huggingface.co/vimassaru/teeth-seg-upernet-convnext-tiny)


## Configurações do Modelo

Para treinar o modelo foi utilizado as seguintes versões dos frameworks:

- Transformers 4.30.2
- Pytorch 2.0.1+cu118
- Datasets 2.13.1
- Tokenizers 0.13.3

## Hiper-parâmetros de treino (utilizando Trainer do Hugging Face)

Foram utilizadas os seguintes argumentos para os parâmetros de treino:

- learning_rate: 0.0006
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 200

## Arquitetura de Rede

A estrutura de rede neural artificial pré treinada utilizada foi o SegFormer **nvidia/mit-b0**. Desenvolvido conforme a imagem abaixo:

<img align="center" src="https://github.com/vimassaru/WIA/blob/main/data/images/segformer_architecture.png">

A implementação envolveu realizar um fine-tuning na última camada do modelo para que ele se adequasse a quantidade de classes das imagens de raio-x.

É possível ver as modificações desenvolvida dentro do jupyter notebook na pasta `src`.

<a href="src/smiledataai_segformer_pretrained.ipynb">Google Colab Notebook</a>

# Reimplementação em TensorFlow + Demo Web

Depois do trabalho original (PyTorch + Hugging Face `transformers`), reimplementei
o treino do SegFormer em **TensorFlow** e publiquei uma página que roda a
inferência **inteiramente no navegador**, via **TensorFlow.js** — sem servidor,
sem backend, sem nada instalado do lado de quem acessa.

🔗 **Testar agora**: [vimassaru.github.io/WIA_Teeth_Semantic_Segmentation](https://vimassaru.github.io/WIA_Teeth_Semantic_Segmentation/)

## Como funciona a versão web

A pasta [`web/`](web/) é uma página estática autocontida (`index.html` +
modelo convertido em `web/model/`), publicada automaticamente no GitHub Pages
a cada push (workflow em [`.github/workflows/deploy-pages.yml`](.github/workflows/deploy-pages.yml)).
Ao abrir a página:

1. O **TensorFlow.js** (carregado via CDN) baixa o modelo já convertido
   (`web/model/model.json` + pesos) e o mantém em memória no navegador.
2. Você arrasta uma imagem de raio-x panorâmico — o pré-processamento
   (redimensionamento, normalização), a inferência e o pós-processamento
   (upsample da máscara e o argmax por pixel) rodam todos dentro do próprio
   grafo do modelo, no seu navegador.
3. O resultado é uma máscara colorida por dente, sobreposta à imagem
   original, com legenda na notação dentária e slider de opacidade.

Nenhuma imagem enviada sai do seu computador — não existe upload pra
servidor algum.

## Treino em TensorFlow

O treino em si (não a inferência) ainda precisa de GPU e roda fora do
navegador, em [`src/tensorflow/`](src/tensorflow/):

- **Modelo**: `keras_hub.models.SegFormerImageSegmenter`, preset
  `segformer_b0_ade20k_512` (mesma família MiT-B0 do artigo original) —
  usado no lugar da `transformers` porque a partir da v5 essa biblioteca
  removeu o suporte a TensorFlow por completo, mantendo só PyTorch.
- **Dataset**: mesmas 113 imagens de raio-x panorâmico do artigo (80 treino
  / 33 teste), com **augmentação de dados** no treino — flip horizontal
  (com remapeamento de quadrante, já que espelhar a imagem troca o lado da
  boca) e variação de brilho/contraste.
- **Otimização**: Adam com *cosine decay* na taxa de aprendizado (em vez de
  taxa fixa), refinando melhor perto da convergência.
- **Ambiente**: WSL2 + Python isolado via `uv`, GPU (RTX 3070) com
  TensorFlow via `tensorflow[and-cuda]`.
- **Exportação**: o modelo treinado (`.keras`) é empacotado com
  pré/pós-processamento embutidos e exportado como SavedModel, depois
  convertido para TensorFlow.js (`tensorflowjs_converter`) — é esse
  artefato final que a página web consome.

Scripts (rodar dentro do venv, na raiz do projeto):

```bash
python src/tensorflow/train.py --epochs 200          # treina e salva o checkpoint
python src/tensorflow/eval_metrics.py                # calcula pixel accuracy / IoU / Dice no teste
python src/tensorflow/export_savedmodel.py            # empacota o modelo de inferência
tensorflowjs_converter --input_format=tf_saved_model data/tensorflow_savedmodel web/model
python src/tensorflow/build_metrics_js.py              # gera web/metrics_data.js pros gráficos da página
```

## Métricas (conjunto de teste)

A própria página web mostra esses números e os gráficos de treino por
época (loss e IoU), comparando o treino com e sem augmentação de dados:

| Métrica | Sem augmentação | Com augmentação (atual) |
|---|---|---|
| IoU médio | 0.747 | **0.760** |
| Test loss final | 0.204 (oscilando) | **0.179** (estável) |
| Pixel accuracy | — | **95.8%** |
| Dice / F1 médio | — | **0.862** |

# Citation

```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
