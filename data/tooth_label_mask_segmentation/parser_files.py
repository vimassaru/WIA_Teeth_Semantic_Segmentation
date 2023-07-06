from pathlib import Path
import shutil

# Caminho da pasta de origem
image_origin = Path('./test')

# Caminho das pastas de destino

image_folder = Path('./test/image')
mask_folder = Path('./test/mask')

# Cria as pastas de destino, se não existirem
image_folder.mkdir(parents=True, exist_ok=True)
mask_folder.mkdir(parents=True, exist_ok=True)

# Percorre a estrutura de diretórios e arquivos da pasta de origem
for item in image_origin.glob('**/*'):
    if item.is_file() and item.suffix.lower() != '.csv':
        if 'mask' in item.name:
            # Caminho de destino para a pasta mask
            dst = mask_folder / item.relative_to(image_origin).name

        else:
            # Caminho de destino para a pasta image
            dst = image_folder / item.relative_to(image_origin).name

        # Verifica se o caminho de origem é diferente do caminho de destino
        if str(item.resolve()) != str(dst.resolve()):
            shutil.move(item, dst)

# Remove a parte "_mask" dos arquivos na pasta mask
for file in mask_folder.glob('*'):
    new_name = file.name.replace('_mask', '')
    file.rename(mask_folder / new_name)


print("Tarefas concluídas com sucesso!")
