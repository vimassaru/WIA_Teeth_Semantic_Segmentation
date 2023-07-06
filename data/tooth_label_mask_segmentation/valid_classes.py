import os


def generate_file_pairs(file_list):
    pairs = []
    for line in file_list:
        line = line.strip()
        image_path, label_path = line.split(' ')
        pairs.append((image_path, label_path))
    return pairs


# Carregar a lista de arquivos de treinamento
with open('list/teeth_dataset/train.lst', 'r') as file:
    file_list = file.readlines()

# Gerar pares de arquivos de imagem e label
pairs = generate_file_pairs(file_list)

# Imprimir os pares de arquivos
for image_path, label_path in pairs:
    print("Imagem:", image_path)
    print("Label:", label_path)
    print()  # Adicionar uma linha vazia para separar os pares

# Resto do seu código...
