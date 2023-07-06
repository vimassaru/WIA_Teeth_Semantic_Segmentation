# WIA
Workshop de Inteligência Artificial - UNIFESP

# Informações do Projeto

Este projeto foi desenvolvido na unidade curricular Inteligência Artificial da Unifesp, onde apliquei conceitos de machine learning para tarefa de segmentação semântica como as redes neurais artificiais e processamento de imagens.

O projeto **SmileDataAI**, é um modelo de aprendizado de máquina para tarefa de segmentação semântica em imagens panorâmicas de raio-x odontológicos. O modelo recebe uma *imagem* como input e retorna em seu output uma predição em formato de *máscara*. Segue abaixo uma predição realizada pelo modelo:

<img align="center" src="https://github.com/vimassaru/WIA/blob/main/data/images/pred_ground_truth.png">

# Tecnologias Utilizadas

Para o desenvolvimento de todo o projeto, foram utilizadas as seguintes tecnologias:
<div style="display: inline_block">
  <img align="center" alt="Icon-Python" height="30" width="40" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg">
  <img align="center" alt="Icon-Jupyter" height="30" width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/jupyter/jupyter-original.svg" />
  <img align="center" alt="Icon-PyTorch" height="30" width="40" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg"/>
</div>


# Pipeline do Projeto

Abaixo está demonstrado o pipeline utilizado para o desenvolvimento da rede neural, destacando as etapas de treino e teste em <span style="color:yellow">amarelo</span> e <span style="color:yellow">amarelo</span>, respectivamente.

<img align="center" src="https://github.com/vimassaru/WIA/blob/main/data/images/SmileDataAI.png">

## Hugging Face - Modelo e Dataset

O meu modelo e o dataset tratado se encontram na página do Hugging Face nos seguintes links:

- [Dataset](https://huggingface.co/datasets/vimassaru/teethsegmentation/tree/main)
- [Modelo](https://huggingface.co/vimassaru/segformer-b0-finetuned-segments-sidewalk-oct-22)


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

A implementação  envolveu realizar um fine-tuning na última camada do modelo para que ele se adequasse a quantidade de classes das imagens de raio-x.

É possível ver as modificações desenvolvida dentro do jupyter notebook na pasta `src`.

<a href="src/smiledataai_segformer_pretrained.ipynb">Google Colab Notebook</a>

# Citation

```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
