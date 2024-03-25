import cv2
import os
from keras.preprocessing.image import ImageDataGenerator

for i in range(1, len(os.listdir('Imagens para treino/cafe'))):
    img = cv2.imread(f"Imagens para treino/cafe/img ({i}).jpg")
    if img is None:
        print(f"Não foi possível ler a imagem img ({i}).jpg")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para o formato correto (RGB)
    img = img.reshape((1,) + img.shape)  # Adiciona uma dimensão extra para o ImageDataGenerator
    
    # Crie um ImageDataGenerator com as transformações desejadas
dataGen = ImageDataGenerator(
    width_shift_range=0.1,   # Altera a posição width da imagem
    height_shift_range=0.1,  # Altera a posição height da imagem
    zoom_range=0.2,           # Aplica zoom
    shear_range=0.1,          # Muda o ângulo
    rotation_range=10         # Rotaciona a imagem
)

# Crie um diretório para salvar as imagens aumentadas, se necessário
output_dir = "Imagens para treino/aumentadas"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
# Use o ImageDataGenerator para gerar imagens aumentadas e salvá-las no mesmo diretório
for i in range(1, 14):
    img = cv2.imread(f"Imagens para treino/cafe/i ({i}).jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte para o formato correto (RGB)
    img = img.reshape((1,) + img.shape)  # Adiciona uma dimensão extra para o ImageDataGenerator

    # Gere imagens aumentadas
    for j, batch in enumerate(dataGen.flow(img, batch_size=1, save_to_dir=output_dir, save_prefix=f"aug_{i}", save_format="png")):
        if j >= 20:  # Defina quantas imagens você deseja gerar
            break
