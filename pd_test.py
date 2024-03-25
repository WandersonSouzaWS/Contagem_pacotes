#--------------------------------------------------------
# Modelo de Classificação realizando detecção pontual   |
# Wanderson Souza - UFV - 2024                          |
#--------------------------------------------------------
#Importando Bicliotecas
import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


##Definindo Funções do projeto
#Desenha area de classificação
def draw_square(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 250, 130), 2)

#Redimencionamento da região
def resize_region(region, target_size):
    return cv2.resize(region, target_size)

#Ajuste da proporção da entrada de video
def resize_keep_aspect_ratio(frame, width):
    ratio = width / frame.shape[1]
    dim = (width, int(frame.shape[0] * ratio))
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


##Carregando dados para o projeto
#Carregando modelo
model = load_model('model/classificacao/modelo_cafe_3.keras')
class_names = []

with open('info/classes.txt', 'r') as arquivos:
    for linha in arquivos:
        classe = linha.strip()
        class_names.append(classe)
        print(classe)

#Carregando entrada camera e ou video
video_path = 'valid/prod.mp4'
cap = cv2.VideoCapture(video_path)


##Definindo parâmetros
#Posição da área de classificação
x = 265         #Posição do ponto no eixo x
y = 180         #Posição do ponto no eixo y
width = 200     #Comprimento da figura
height = 260    #Altura da figura

#Tamanho fixo
fixed_width = 800

#Contagem de pacotes
contador_pacotes = 0
status_anterior = "nada"
taxa1 = 1; taxa2 = 1;t1 = 1; t2 =1

##Inicia rotina while
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    #Redimenssionamento da entrada
    resized_frame = resize_keep_aspect_ratio(frame, fixed_width)
    #Extração da região
    region_of_interest = resized_frame[y:y+height, x+100:(x+100+width-100)] #ATENÇÃO
    #Redimenssionamento da região para tamanho do modelo
    resized_region = resize_region(region_of_interest, (200, 260)) #ATENÇÃO
    
    #------------------------------------------------------------------------
    #Inicia contagem de tempo para FPS
    start = time.time()
    
    # Classificar a região delimitada usando o modelo classificador  
    img_array = np.expand_dims(resized_region, axis=0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0]) 
    print(class_names[np.argmax(score)])      
    #Realizando predição
    if class_names[np.argmax(score)] == "pacote_cinza":
        
        if np.max(score) > 0.9:    
            if class_names[np.argmax(score)] == "pacote_cinza" and status_anterior == "nada":
                contador_pacotes += 1
                print("Imagem identificada. \nN°: ",contador_pacotes)
                status_anterior = "identificado" #trava a contagem enquanto a imagem estiver identificada            
                #Taxa de produção
                if taxa1 == taxa2:
                    t1 = time.time()
                    taxa1 = contador_pacotes
                else: 
                    t2 = time.time()
                    taxa2 = (1800/(t2-t1))
                    taxa1 = taxa2
                                
    else: 
        status_anterior = "nada"
                        
    #Finaliza contagem de tempo para FPS
    end = time.time()
    #------------------------------------------------------------------------
     
    # Ajusta texto FPS e contagem de pacotes
    fps_label = f"FPS: {round((1.0/(end - start)), 2)}"
    cont_pac = f"Qte. Pacotes: {contador_pacotes}"
    taxa3 = "Taxa prod.: {:.2f} Kg/h".format(taxa2)
    # Escrevendo no video
    cv2.putText(resized_frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 130), 2)
    cv2.putText(resized_frame, taxa3, (265, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 130), 2)
    cv2.putText(resized_frame, cont_pac, (265, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 250, 130), 2)

     # Desenhar o quadrado no frame redimensionado
    draw_square(resized_frame, x, y, width, height)
    draw_square(resized_frame, x+100, y, width-100, height)
    
    # Exibir o frame redimensionado com o quadrado desenhado
    cv2.imshow('Frame', resized_frame)
    
    # Esperar pela tecla 'q' para sair do loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
