from pathlib import Path


def create_lst_file(folder_path, lst_file_path):
    folder_path = Path(folder_path)
    lst_file_path = Path(lst_file_path)

    folder_google_colab = 'teeth_dataset/train/image/'

    # Obtém a lista de arquivos na pasta
    files = list(folder_path.glob('*'))

    # Cria o arquivo .lst e escreve os nomes dos arquivos
    with open(lst_file_path, 'w') as lst_file:
        for file in files:
            lst_file.write(f"{folder_google_colab}{file.name}\n")

    print(f"Arquivo .lst criado: {lst_file_path}")


# Caminho da pasta
folder_path = './train/image'

# Caminho do arquivo .lst
lst_file_path = './list/teeth_dataset/train_image_list.lst'

create_lst_file(folder_path, lst_file_path)
