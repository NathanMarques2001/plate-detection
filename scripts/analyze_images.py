import cv2
import numpy as np
import os
from ultralytics import YOLO

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
currentDir = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(currentDir)  # sobe 1 nível
IMAGE_PATH = os.path.join(root, "media", "frame-1.png")
OUTPUT_DIR = 'academic_results'
MODEL_PATH = os.path.join(root, "models", "license_plate_detector.pt")

# Garante que a pasta de saída existe
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def apply_academic_processing(plate_img, save_prefix):
    """
    Aplica o pipeline de pré-processamento e salva os resultados intermediários
    para o relatório.
    """
    print(f"Processando placa detectada...")

    # 1. UPSCALING (Interpolação Cúbica)
    # Aumenta a resolução para melhorar a detecção de bordas
    h, w = plate_img.shape[:2]
    img_resized = cv2.resize(plate_img, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'{OUTPUT_DIR}/{save_prefix}_1_upscaled.jpg', img_resized)

    # 2. ESCALA DE CINZA
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{OUTPUT_DIR}/{save_prefix}_2_gray.jpg', gray)

    # 3. FILTRO BILATERAL (Blur Analysis)
    # Remove ruído preservando bordas (melhor que Gaussian)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    cv2.imwrite(f'{OUTPUT_DIR}/{save_prefix}_3_bilateral_blur.jpg', blur)

    # ---------------------------------------------------------
    # REQUISITO 1: CANNY EDGE DETECTION
    # ---------------------------------------------------------
    edges = cv2.Canny(blur, 100, 200)
    cv2.imwrite(f'{OUTPUT_DIR}/{save_prefix}_4_canny_edges.jpg', edges)
    print(f" - Saved: {save_prefix}_4_canny_edges.jpg")

    # ---------------------------------------------------------
    # REQUISITO 2: HARRIS CORNER DETECTION
    # ---------------------------------------------------------
    gray_float = np.float32(blur)
    # Detector de Harris (block_size=2, ksize=3, k=0.04)
    dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
    dst = cv2.dilate(dst, None) # Dilata para visualizar melhor
    
    # Gera imagem visual (pontos vermelhos sobre a imagem original)
    harris_img = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
    harris_img[dst > 0.01 * dst.max()] = [0, 0, 255] # Pontos vermelhos
    
    cv2.imwrite(f'{OUTPUT_DIR}/{save_prefix}_5_harris_corners.jpg', harris_img)
    print(f" - Saved: {save_prefix}_5_harris_corners.jpg")

    # ---------------------------------------------------------
    # PREPARAÇÃO PARA OCR (Limiarização)
    # ---------------------------------------------------------
    # Sharpening para destacar letras antes do threshold
    kernel_sharpening = np.array([[-1, -1, -1], 
                                  [-1,  9, -1],
                                  [-1, -1, -1]])
    sharp = cv2.filter2D(blur, -1, kernel_sharpening)
    
    # Otsu Thresholding (Binarização Automática)
    _, thresh = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f'{OUTPUT_DIR}/{save_prefix}_6_otsu_threshold.jpg', thresh)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
def main():
    print("--- GERADOR DE IMAGENS (CANNY/HARRIS) ---")
    
    # 1. Carregar Imagem
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Erro: Não foi possível abrir a imagem {IMAGE_PATH}")
        return

    # 2. Carregar Modelo de Detecção de Placas
    print("Carregando modelo YOLO...")
    model = YOLO(MODEL_PATH)

    # 3. Detectar Placas
    results = model(img)[0]

    if len(results.boxes.data.tolist()) == 0:
        print("Nenhuma placa detectada na imagem.")
        return

    # 4. Processar cada placa encontrada
    for i, detection in enumerate(results.boxes.data.tolist()):
        x1, y1, x2, y2, score, class_id = detection
        
        # Recortar a placa (Crop)
        plate_crop = img[int(y1):int(y2), int(x1):int(x2), :]
        
        # Salvar o crop original para referência
        cv2.imwrite(f'{OUTPUT_DIR}/plate_{i}_original.jpg', plate_crop)
        
        # Aplicar os filtros acadêmicos
        apply_academic_processing(plate_crop, save_prefix=f"plate_{i}")

    print(f"\nConcluído! Verifique a pasta '{OUTPUT_DIR}' para ver os resultados.")

if __name__ == "__main__":
    main()