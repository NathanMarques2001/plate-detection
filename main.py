# main.py
"""
Orquestrador do pipeline completo:

1. Detecção e OCR de placas -> data/result.csv
2. Interpolação de bounding boxes -> data/result-interpolated.csv
3. Geração do vídeo final -> media/video-final.mp4
"""

from scripts.object_identifier import run_object_identifier
from scripts.interpolate_data import run_interpolation
from scripts.video_writer import write_video

if __name__ == "__main__":
    print("=== Etapa 1: Detectando veículos e placas (YOLO + SORT + OCR) ===")
    run_object_identifier()

    print("\n=== Etapa 2: Interpolando bounding boxes ===")
    run_interpolation()

    print("\n=== Etapa 3: Gerando vídeo final ===")
    write_video()

    print("\nPipeline concluído com sucesso! ✅")
