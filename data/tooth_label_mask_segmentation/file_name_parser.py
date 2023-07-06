from pathlib import Path


def remove_text_from_filenames(folder_path, text):
    folder_path = Path(folder_path)

    # Percorre todos os arquivos e subpastas na pasta especificada
    for item in folder_path.glob('**/*'):
        if item.is_file():
            new_filename = item.name.replace(text, "")
            new_file_path = item.with_name(new_filename)

            # Renomeia o arquivo
            item.rename(new_file_path)


# Template para remover trechos do nome do arquivo
folder_path = "./test"  # Caminho da pasta onde os arquivos estão
text_to_remove = ".rf."  # Trecho a ser removido dos nomes dos arquivos

remove_text_from_filenames(folder_path, text_to_remove)

print('arquivos renomeados com sucesso!')
