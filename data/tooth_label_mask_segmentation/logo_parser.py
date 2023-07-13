import csv
import numpy as np
import os
from PIL import Image
from pathlib import Path


def load_color_mapping(file_path):
    color_mapping = {}
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Ignorar o cabeçalho do CSV
        for row in csv_reader:
            class_value = int(row[0])
            # Remove espaços em branco antes e depois do nome da classe
            class_name = row[1].strip()
            color_mapping[class_value] = class_name
    return color_mapping


def colorize_classes(image_path, output_path, color_mapping):
    # Abre a imagem em escala de cinza
    image = Image.open(image_path).convert("L")

    # Converte a imagem em um array NumPy
    image_array = np.array(image)

    # Cria uma matriz vazia para a imagem colorizada
    colored_image = np.zeros(
        (image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    # Preenche a matriz com as cores mapeadas para cada classe
    for class_value, class_name in color_mapping.items():
        color = class_colors[class_name]
        colored_image[image_array == class_value] = color

    # Cria uma nova imagem a partir da matriz colorizada
    colored_image = Image.fromarray(colored_image)

    # Salva a imagem colorizada em formato PNG
    colored_image.save(output_path)

    print(f"Imagem colorizada salva em '{output_path}'.")


# Definição das cores para cada classe
class_colors = {
    'background': (0, 0, 0),
    '11': (255, 0, 0),
    '12': (0, 255, 0),
    '13': (0, 0, 255),
    '14': (255, 255, 0),
    '15': (255, 0, 255),
    '16': (0, 255, 255),
    '17': (255, 128, 0),
    '18': (128, 0, 255),
    '21': (0, 255, 128),
    '22': (128, 255, 0),
    '23': (0, 128, 255),
    '24': (255, 0, 128),
    '25': (128, 255, 128),
    '26': (128, 128, 255),
    '27': (255, 128, 128),
    '28': (128, 128, 128),
    '31': (0, 0, 128),
    '32': (0, 128, 0),
    '33': (128, 0, 0),
    '34': (0, 128, 128),
    '35': (128, 0, 128),
    '36': (128, 128, 0),
    '37': (128, 64, 0),
    '38': (64, 128, 0),
    '41': (0, 64, 128),
    '42': (64, 0, 128),
    '43': (128, 128, 64),
    '44': (64, 64, 128),
    '45': (128, 64, 64),
    '46': (64, 128, 64),
    '47': (128, 128, 64),
    '48': (64, 128, 128),
}
# Caminho da imagem em escala de cinza
image_path = Path('train/mask/100_png23a801b2d65aa55e991be54db935ebe2.png')

# Caminho do arquivo CSV com o mapeamento de classes
csv_path = Path('train/_classes.csv')

# Carrega o mapeamento de classes e cores do arquivo CSV
color_mapping = load_color_mapping(csv_path)

# Caminho de saída para a imagem colorizada
output_path = Path('output/heat_map.png')

# Chama a função para colorizar as classes na imagem
# # colorize_classes(image_path, output_path, color_mapping)

# ---------------------------------------------------------------------- #


def combine_images(image1, image2):
    # Abre as imagens
    img1 = Image.open(image1)
    img2 = Image.open(image2)

    # Obtém a largura e altura das imagens
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Divide as imagens ao meio
    half_width1 = width1 // 2
    half_width2 = width2 // 2

    left_half_img1 = img1.crop((0, 0, half_width1, height1))
    right_half_img2 = img2.crop((half_width2, 0, width2, height2))

    left_half_img2 = img2.crop((0, 0, half_width2, height2))
    right_half_img1 = img1.crop((half_width1, 0, width1, height1))

    # Combina as metades das imagens
    combined_img1 = Image.new('RGB', (width1, height1))
    combined_img1.paste(left_half_img1, (0, 0))
    combined_img1.paste(right_half_img2, (half_width1, 0))

    combined_img2 = Image.new('RGB', (width2, height2))
    combined_img2.paste(left_half_img2, (0, 0))
    combined_img2.paste(right_half_img1, (half_width2, 0))

    # Cria a pasta "output" se não existir
    output_folder = Path("output")
    output_folder.mkdir(exist_ok=True)

    # Salva as novas imagens na pasta "output"
    combined_img1_path = output_folder / (image1.stem + "_combined.jpg")
    combined_img2_path = output_folder / (image2.stem + "_combined.jpg")
    combined_img1.save(combined_img1_path)
    combined_img2.save(combined_img2_path)

    print(
        f"Imagens combinadas foram salvas em '{output_folder}' como '{combined_img1_path.name}' e '{combined_img2_path.name}'.")


# Caminhos das imagens de entrada
image1_path = Path('train/image/100_png23a801b2d65aa55e991be54db935ebe2.jpg')
image2_path = Path('output/heat_map.png')

# Chama a função para combinar as imagens
# combine_images(image1_path, image2_path)

# ---------------------------------------------------------------------- #


def colorize_images_folder(input_folder, output_folder, color_mapping):
    # Verifica se a pasta de saída existe e a cria, se necessário
    os.makedirs(output_folder, exist_ok=True)

    # Percorre todas as imagens na pasta de entrada
    for image_file in os.listdir(input_folder):
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            # Caminho completo para a imagem de entrada
            input_image_path = os.path.join(input_folder, image_file)

            # Cria o nome de saída para a imagem colorizada
            output_image_file = os.path.splitext(
                image_file)[0] + "_colorized.png"
            output_image_path = os.path.join(output_folder, output_image_file)

            # Chama a função colorize_classes para colorizar a imagem e salvá-la
            colorize_classes(input_image_path,
                             output_image_path, color_mapping)

            print(f"Imagem colorizada salva em '{output_image_path}'.")


# Caminho da pasta de entrada contendo as imagens em escala de cinza
input_folder = Path('train/mask')

# Caminho da pasta de saída para as imagens colorizadas
output_folder = Path('output/colorized')

# Chama a função para colorizar todas as imagens na pasta
colorize_images_folder(input_folder, output_folder, color_mapping)
