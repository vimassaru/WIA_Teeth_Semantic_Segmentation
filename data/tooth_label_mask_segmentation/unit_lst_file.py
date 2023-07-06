import os


def unificar_arquivos(arquivo1, arquivo2, arquivo_saida):
    with open(arquivo1, 'r') as file1, open(arquivo2, 'r') as file2, open(arquivo_saida, 'w') as output_file:
        caminhos1 = file1.read().splitlines()
        caminhos2 = file2.read().splitlines()

        for caminho1 in caminhos1:
            nome_arquivo1 = os.path.basename(caminho1)
            nome_base1 = os.path.splitext(nome_arquivo1)[0]

            # Busca o caminho correspondente no arquivo 2 com base no nome do arquivo
            caminho2 = next((caminho for caminho in caminhos2 if os.path.splitext(
                os.path.basename(caminho))[0] == nome_base1), '')

            linha = f'{caminho1} {caminho2}\n'
            output_file.write(linha)

    print("Arquivos unificados com sucesso!")


# Formas de uso:
arquivo1 = './list/teeth_dataset/train_image_list.lst'
arquivo2 = './list/teeth_dataset/train_label_list.lst'
arquivo_saida = './list/teeth_dataset/train.lst'

unificar_arquivos(arquivo1, arquivo2, arquivo_saida)
