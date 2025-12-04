import cv2
import numpy as np


def preprocess_plate(img, car_id, frame_nmr):
    """
    Pré-processamento de um recorte de placa para facilitar OCR.

    Etapas aplicadas:
    1) Upscaling (interpolação cúbica) para aumentar resolução de placas pequenas.
    2) Conversão para escala de cinza.
    3) Filtro bilateral para remover ruído preservando bordas.
    4) Sharpening (filtro de nitidez) para realçar contornos de caracteres.
    5) Thresholding com Otsu para binarizar a imagem (preparação para OCR).

    Parâmetros:
    - img: imagem BGR (numpy array) contendo a placa recortada
    - car_id, frame_nmr: apenas para referência/debug (não usados internamente)

    Retorna:
    - `thresh`: imagem binarizada pronta para OCR (numpy array)
    """

    # 1. UPSCALING: aumenta a resolução para melhorar a legibilidade de caracteres
    height, width = img.shape[:2]
    img_resized = cv2.resize(img, (width * 3, height * 3), interpolation=cv2.INTER_CUBIC)

    # 2. CONVERSÃO PARA ESCALA DE CINZA: OCR normalmente opera sobre imagens em escala de cinza
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 3. FILTRO BILATERAL: remove ruído preservando as bordas (útil para manter traços de caracteres)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    # 4. SHARPENING: realça bordas e traços para aumentar contraste dos caracteres
    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharp = cv2.filter2D(blur, -1, kernel_sharpening)

    # 5. THRESHOLD (Otsu): binariza a imagem; o sinal invertido (`THRESH_BINARY_INV`) pode
    # ser útil dependendo do contraste entre caracteres e fundo da placa
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh