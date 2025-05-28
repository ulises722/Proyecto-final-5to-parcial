# Importar librerias
import cv2
import mediapipe as mp

# Función para calcular la distancia euclidiana entre dos puntos
def distancia_euclidiana(p1, p2):
    d = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return d

# Funcion para dibujar un bounding box alrededor de la mano
def draw_bounding_box(image, hand_landmarks):
    image_height, image_width, _ = image.shape
    x_min, y_min = image_width, image_height
    x_max, y_max = 0, 0
    
    # Iterar entre los landmarks de la mano para encontrar los limites del bounding box
    for landmark in hand_landmarks.landmark:
        x, y = int(landmark.x * image_width), int(landmark.y * image_height)
        if x < x_min: x_min = x
        if y < y_min: y_min = y
        if x > x_max: x_max = x
        if y > y_max: y_max = y
    
    # Dibujar el bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

# Vaiables para detectar la mano y darle estilo
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Inicializar la cámara
cap = cv2.VideoCapture(0)
cap.set(3,1920)
cap.set(4,1080)

# Inicializar MediaPipe Hands
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
  
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      continue

    # Convertir las imagenes de BGR a RGB y espejearlas
    # MediaPipe requiere que la imagen sea RGB 
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    # Si se detecta una mano, se dibujan los landmarks y las conexiones
    if results.multi_hand_landmarks:
        if len(results.multi_hand_landmarks):
            for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                
                # Dibjujar los landmarks y las conexiones
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                
                # Funcion para dibujar el bounding box de la mano
                draw_bounding_box(image, hand_landmarks)

                # Obtener las coordenadas de los puntos clave de la mano
                # Dedo índice parte superior
                index_finger_tip = (int(hand_landmarks.landmark[8].x * image_width),
                                int(hand_landmarks.landmark[8].y * image_height))
                # Dedo índice parte inferior
                index_finger_pip = (int(hand_landmarks.landmark[6].x * image_width),
                                int(hand_landmarks.landmark[6].y * image_height))
                
                # Dedo pulgar parte superior
                thumb_tip = (int(hand_landmarks.landmark[4].x * image_width),
                                int(hand_landmarks.landmark[4].y * image_height))
                # Dedo pulgar parte inferior
                thumb_pip = (int(hand_landmarks.landmark[2].x * image_width),
                                int(hand_landmarks.landmark[2].y * image_height))
                
                # Dedo medio parte superior
                middle_finger_tip = (int(hand_landmarks.landmark[12].x * image_width),
                                int(hand_landmarks.landmark[12].y * image_height))
                # Dedo medio parte inferior
                middle_finger_pip = (int(hand_landmarks.landmark[10].x * image_width),
                                int(hand_landmarks.landmark[10].y * image_height))
                
                # Dedo anular parte superior
                ring_finger_tip = (int(hand_landmarks.landmark[16].x * image_width),
                                int(hand_landmarks.landmark[16].y * image_height))
                # Dedo anular parte media
                ring_finger_pip = (int(hand_landmarks.landmark[14].x * image_width),
                                int(hand_landmarks.landmark[14].y * image_height))
                
                # Dedo meñique parte superior
                pinky_tip = (int(hand_landmarks.landmark[20].x * image_width),
                                int(hand_landmarks.landmark[20].y * image_height))
                # Dedo meñique parte inferior
                pinky_pip = (int(hand_landmarks.landmark[18].x * image_width),
                                int(hand_landmarks.landmark[18].y * image_height))
                
                # Muñeca
                wrist = (int(hand_landmarks.landmark[0].x * image_width),
                                int(hand_landmarks.landmark[0].y * image_height))
                
                # Dedo anular parte inferior 
                ring_finger_pip2 = (int(hand_landmarks.landmark[5].x * image_width),
                                int(hand_landmarks.landmark[5].y * image_height))
                


                # LETRAS
                # Calulcar la distancia entre los puntos clave para determinar que letra se esta mostrando
                # A
                if index_finger_pip[1] < index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and abs(distancia_euclidiana(index_finger_tip, thumb_tip)) > 120:

                #if abs(thumb_tip[1] - index_finger_pip[1]) <45 \
                #    and abs(thumb_tip[1] - middle_finger_pip[1]) < 30 and abs(thumb_tip[1] - ring_finger_pip[1]) < 30\
                #    and abs(thumb_tip[1] - pinky_pip[1]) < 30:
                    
                    cv2.putText(image, 'A', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # B  
                # pip > tip = dedo extendido
                # pip < tip = dedo doblado
                elif index_finger_pip[1] > index_finger_tip[1] and pinky_pip[1] > pinky_tip[1] and \
                    middle_finger_pip[1] > middle_finger_tip[1] and ring_finger_pip[1] > ring_finger_tip[1] and \
                        middle_finger_tip[1] < ring_finger_tip[1] and distancia_euclidiana(thumb_tip, pinky_pip) <100:
                    
                    cv2.putText(image, 'B', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                
                # C    
                elif index_finger_tip[1] < middle_finger_pip[1] and index_finger_tip[1] < middle_finger_tip[1] and \
                index_finger_tip[1] > index_finger_pip[1] and abs(index_finger_tip[1] - thumb_tip[1]) < 400 and abs(index_finger_tip[1] - thumb_tip[1]) > 110 \
                and middle_finger_pip[1] < middle_finger_tip[1] and ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1]:
                   # Indice por encima de los otros dedos
                   cv2.putText(image, 'C', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                
                # D
                elif distancia_euclidiana(thumb_tip, middle_finger_tip) < 65 \
                    and distancia_euclidiana(thumb_tip, ring_finger_tip) < 65 \
                    and  pinky_pip[1] < pinky_tip[1]\
                    and index_finger_pip[1] > index_finger_tip[1]:

                    cv2.putText(image, 'D', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # E   
                elif index_finger_pip[1] < index_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                middle_finger_pip[1] < middle_finger_tip[1] and ring_finger_pip[1] < ring_finger_tip[1] \
                and  abs(index_finger_tip[1] - thumb_tip[1]) > 45 and thumb_tip[1] > index_finger_tip[1] and thumb_tip[1] > middle_finger_tip[1]\
                and thumb_tip[1] > ring_finger_tip[1] and thumb_tip[1] > pinky_tip[1]:
                    # Pulgar por debajo de los otros dedos
                    cv2.putText(image, 'E', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                    
                # F
                # pip > tip = dedo extendido
                # pip < tip = dedo doblado    
                elif  pinky_pip[1] > pinky_tip[1] and middle_finger_pip[1] > middle_finger_tip[1] and \
                    ring_finger_pip[1] > ring_finger_tip[1] and index_finger_pip[1] < index_finger_tip[1]\
                        and abs(thumb_pip[1] > thumb_tip[1]) and distancia_euclidiana(index_finger_tip, thumb_tip) <65:

                    cv2.putText(image, 'F', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # G (Solo funciona con la mano derecha)
                elif index_finger_pip[0] > index_finger_tip[0] and middle_finger_pip[0] < middle_finger_tip[0] and \
                ring_finger_pip[0] < ring_finger_tip[0] and pinky_pip[0] < pinky_tip[0] and \
                thumb_pip[1] > thumb_tip[1]:
                    
                    cv2.putText(image, 'G', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # H
                elif index_finger_pip[0] > index_finger_tip[0] and middle_finger_pip[0] > middle_finger_tip[0] and \
                ring_finger_pip[0] < ring_finger_tip[0] and pinky_pip[0] < pinky_tip[0] and \
                thumb_pip[1] > thumb_tip[1]:
                    
                    cv2.putText(image, 'H', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # I
                # pip > tip = dedo extendido
                # pip < tip = dedo doblado 
                elif index_finger_pip[1] < index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] > pinky_tip[1] and \
                thumb_tip[1] < index_finger_tip[1] and thumb_tip[1] < middle_finger_tip[1] and thumb_tip[1] < ring_finger_tip[1]:  
                # Pulgar por encima de los dedos
                    cv2.putText(image, 'I', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)    
                
                # J 
                elif index_finger_pip[0] < index_finger_tip[0] and middle_finger_pip[0] < middle_finger_tip[0] and \
                ring_finger_pip[0] < ring_finger_tip[0] and pinky_pip[0] > pinky_tip[0] and distancia_euclidiana(index_finger_tip, thumb_tip) < 100:
                #thumb_tip[1] > index_finger_tip[1] and thumb_tip[1] > middle_finger_tip[1] and thumb_tip[1] > ring_finger_tip[1]:
                # Pulgar por encima de los dedos
                    cv2.putText(image, 'J', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)    
                

                # K (Mano derecha)
                elif index_finger_pip[0] > index_finger_tip[0] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[0] < ring_finger_tip[0] and pinky_pip[0] < pinky_tip[0] and \
                thumb_pip[0] > thumb_tip[0] and thumb_pip[1] > index_finger_pip[1] and thumb_pip[1] < middle_finger_pip[1] and distancia_euclidiana(index_finger_pip, thumb_tip) > 10:
                    
                    cv2.putText(image, 'K', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)    
                
                # L (mano derecha)
                # pip > tip = dedo extendido
                # pip < tip = dedo doblado 
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[0] > thumb_tip[0]:
                    
                    cv2.putText(image, 'L', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # M
                elif index_finger_pip[1] < index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[1] < thumb_tip[1]:
                    
                    cv2.putText(image, 'M', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # N
                elif index_finger_pip[1] < index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] > ring_finger_tip[1] and pinky_pip[1] > pinky_tip[1] and \
                thumb_pip[0] < thumb_tip[0] and distancia_euclidiana(middle_finger_pip, thumb_tip) > 10:
                    
                    cv2.putText(image, 'N', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # O
                elif index_finger_pip[1] < index_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                middle_finger_pip[1] < middle_finger_tip[1] and ring_finger_pip[1] < ring_finger_tip[1] \
                and abs(index_finger_tip[1] - thumb_tip[1]) < 50 and thumb_tip[1] > index_finger_tip[1] and thumb_tip[1] > middle_finger_tip[1]\
                and thumb_tip[1] > ring_finger_tip[1] and thumb_tip[1] > pinky_tip[1]:
                    # Pulgar por debajo de los otros dedos
                    cv2.putText(image, 'O', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                
                # P
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] > middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[1] > thumb_tip[1] and abs(distancia_euclidiana(index_finger_tip, middle_finger_tip)) > 120 and \
                    abs(distancia_euclidiana(index_finger_pip, thumb_tip)) < 100:
                    
                    cv2.putText(image, 'P', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                
                # Q
                elif index_finger_pip[0] > index_finger_tip[0] and middle_finger_pip[0] < middle_finger_tip[0] and \
                ring_finger_pip[0] < ring_finger_tip[0] and pinky_pip[0] < pinky_tip[0] and \
                thumb_pip[0] > thumb_tip[0]:
                    
                    cv2.putText(image, 'Q', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # R
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] > middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[1] > thumb_tip[1] and abs(distancia_euclidiana(index_finger_tip, middle_finger_tip)) > 5 and \
                abs(distancia_euclidiana(index_finger_tip, middle_finger_tip)) < 50:
                    
                    cv2.putText(image, 'R', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                
                # S
                elif index_finger_pip[1] < index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[0] < thumb_tip[0]: 
    
                    cv2.putText(image, 'S', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # T
                elif index_finger_pip[1] < index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[1] > thumb_tip[1] and abs(distancia_euclidiana(middle_finger_pip, thumb_tip)) > 40: 
                    
                    cv2.putText(image, 'T', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # U
                # pip > tip = dedo extendido
                # pip < tip = dedo doblado
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] > middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[1] > thumb_tip[1] and abs(distancia_euclidiana(index_finger_tip, middle_finger_tip)) > 50 and \
                abs(distancia_euclidiana(index_finger_tip, middle_finger_tip)) < 100:
                    
                    cv2.putText(image, 'U', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # V
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] > middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and \
                thumb_pip[1] > thumb_tip[1] and abs(distancia_euclidiana(index_finger_tip, middle_finger_tip)) > 100:
                    
                    cv2.putText(image, 'V', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # W
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] > middle_finger_tip[1] and \
                ring_finger_pip[1] > ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and abs(distancia_euclidiana(pinky_tip, thumb_tip)) < 50:
                    
                    cv2.putText(image, 'W', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # X 
                elif abs(distancia_euclidiana(index_finger_tip, thumb_tip)) > 80 and \
                pinky_pip[0] > pinky_tip[0] and ring_finger_pip[0] > ring_finger_tip[0] and middle_finger_pip[0] > middle_finger_tip[0]:
                    
                    cv2.putText(image, 'X', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

                # Y
                # pip > tip = dedo extendido
                # pip < tip = dedo doblado
                elif index_finger_pip[0] < index_finger_tip[0] and middle_finger_pip[0] < middle_finger_tip[0] and \
                ring_finger_pip[0] < ring_finger_tip[0] and pinky_pip[0] > pinky_tip[0] and thumb_pip[1] > thumb_tip[1]:
                    
                    cv2.putText(image, 'Y', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)
                
                # Z
                elif index_finger_pip[1] > index_finger_tip[1] and middle_finger_pip[1] < middle_finger_tip[1] and \
                ring_finger_pip[1] < ring_finger_tip[1] and pinky_pip[1] < pinky_tip[1] and abs(distancia_euclidiana(index_finger_pip, thumb_tip)) < 100 \
                and abs(distancia_euclidiana(middle_finger_pip, thumb_tip)) < 70:
                # Pulgar por encima de los dedos
                    cv2.putText(image, 'Z', (700, 150), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 255), 6)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()