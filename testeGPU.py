import subprocess
import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt

model = load_model('meu_modelo.keras')

# Executa o script YOLOv5
subprocess.run(['python', 'yolov5/detect.py'])

def classificar_imagem(img_path):
    # Carregue a imagem e redimensione para o tamanho de entrada do modelo
    img = image.load_img(img_path, target_size=(224, 224))  # Ajuste o tamanho conforme necessário
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalização se necessário

    # Faça a previsão
    previsao = model.predict(img_array)
    categoria = np.argmax(previsao, axis=1)

    return categoria

categorias = {0: 'Gato', 1: 'Cachorro'}

def exibir_imagem_com_categoria(img_path):
    categoria_idx = classificar_imagem(img_path)
    categoria_nome = categorias[categoria_idx[0]]

    # Exibir a imagem
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Categoria: {categoria_nome}')
    plt.show()

# Função para processar todas as imagens em uma pasta
def processar_imagens_da_pasta(pasta):
    for arquivo in os.listdir(pasta):
        # Verifica se o arquivo é uma imagem (ajuste as extensões conforme necessário)
        if arquivo.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(pasta, arquivo)
            exibir_imagem_com_categoria(img_path)

# Chame a função para processar as imagens na pasta desejada
processar_imagens_da_pasta('imagemTeste')
