"""
Utilitários para leitura e formatação de placas.

Este módulo encapsula o leitor OCR (easyocr) e funções auxiliares para:
- gravar resultados em CSV;
- validar o formato da placa lida;
- aplicar mapeamentos comuns entre caracteres confundidos (ex: 'O' <-> '0').

As funcionalidades aqui são usadas pelo fluxo principal para extrair texto
de crops de placas e normalizar possíveis confusões entre caracteres.
"""

import string
import easyocr

# Inicializa o leitor do EasyOCR para o idioma inglês. GPU está desabilitada
# por padrão para compatibilidade; habilite com gpu=True se tiver GPU e drivers.
reader = easyocr.Reader(['en'], gpu=False)

# Mapas para corrigir confusões comuns entre letras e dígitos
# Por exemplo, OCR pode reconhecer 'O' quando o correto é o dígito '0'.
dict_char_to_int = {'O': '0',
                    'I': '1',
                    'J': '3',
                    'A': '4',
                    'G': '6',
                    'S': '5'}

dict_int_to_char = {'0': 'O',
                    '1': 'I',
                    '3': 'J',
                    '4': 'A',
                    '6': 'G',
                    '5': 'S'}


def write_csv(results, output_path):
    """
    Grava os resultados em um arquivo CSV.

    Args:
        results (dict): Dicionário contendo os resultados.
        output_path (str): Caminho do arquivo CSV de saída.
    """
    # Abre o arquivo de saída e escreve um cabeçalho fixo
    with open(output_path, 'w') as f:
        f.write('{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                'license_plate_bbox', 'license_plate_bbox_score', 'license_number',
                                                'license_number_score'))

        # Itera pelos frames e veículos no dicionário `results`
        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                # Impressão de debug (pode ser removida ou desabilitada)
                print(results[frame_nmr][car_id])
                # Checa se os campos essenciais existem antes de escrever
                if 'car' in results[frame_nmr][car_id].keys() and \
                   'license_plate' in results[frame_nmr][car_id].keys() and \
                   'text' in results[frame_nmr][car_id]['license_plate'].keys():
                    f.write('{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                            car_id,
                                                            # Converte bbox do carro para string no mesmo formato do resto do projeto
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['car']['bbox'][0],
                                                                results[frame_nmr][car_id]['car']['bbox'][1],
                                                                results[frame_nmr][car_id]['car']['bbox'][2],
                                                                results[frame_nmr][car_id]['car']['bbox'][3]),
                                                            # Converte bbox da placa
                                                            '[{} {} {} {}]'.format(
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                            # Score do bbox da placa e texto + score do texto
                                                            results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                            results[frame_nmr][car_id]['license_plate']['text'],
                                                            results[frame_nmr][car_id]['license_plate']['text_score'])
                            )


def license_complies_format(text):
    """
    Verifica se um texto possui o formato esperado de placa.

    Regras aplicadas (formato esperado digital/alfanumérico, 7 caracteres):
    - posições 0,1,4,5,6: letras (ou dígitos que podem mapear para letras)
    - posições 2,3: dígitos (ou letras que podem mapear para dígitos)
    """
    if len(text) != 7:
        return False

    # Checagens por posição, aceitando também os caracteres que podem ser mapeados
    if (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys()) and \
       (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys()) and \
       (text[2] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[2] in dict_char_to_int.keys()) and \
       (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in dict_char_to_int.keys()) and \
       (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys()) and \
       (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys()) and \
       (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys()):
        return True
    else:
        return False


def format_license(text):
    """
    Aplica mapeamentos posicionais para normalizar a placa.

    Algumas posições tendem a ser letras (mas OCR retorna dígitos) e outras
    tendem a ser dígitos (mas OCR retorna letras). O dicionário `mapping`
    define qual mapa aplicar em cada posição.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in [0, 1, 2, 3, 4, 5, 6]:
        if text[j] in mapping[j].keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]

    return license_plate_


def read_license_plate(license_plate_crop):
    """
    Lê o texto da placa a partir da imagem recortada fornecida.

    Args:
        license_plate_crop (PIL.Image.Image): Imagem recortada contendo a placa.

    Returns:
        tuple: Tupla contendo o texto da placa formatado e seu score de confiança.
    """

    # Usa o EasyOCR para ler o crop da placa.
    # `reader.readtext` retorna uma lista de tuples: (bbox, texto, score)
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection

        # Normaliza o texto para maiúsculas e remove espaços
        text = text.upper().replace(' ', '')

        # Se o texto cumprir o formato esperado de placa, aplica os mapeamentos
        if license_complies_format(text):
            return format_license(text), score

    # Se nada válido for encontrado, retorna None
    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Recupera as coordenadas e ID do veículo baseado nas coordenadas da placa.

    Args:
        license_plate (tuple): Tupla contendo as coordenadas da placa (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): Lista de IDs de rastreamento de veículos e suas coordenadas.

    Returns:
        tuple: Tupla contendo as coordenadas do veículo (x1, y1, x2, y2) e ID.
    """
    # Recebe as coordenadas da placa detectada e tenta achar o carro correspondente
    x1, y1, x2, y2, score, class_id = license_plate

    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Se o bbox da placa estiver inteiramente contido no bbox do carro,
        # assumimos que essa placa pertence a este carro.
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break

    if foundIt:
        return vehicle_track_ids[car_indx]

    # Caso não encontre correspondência, retorna indicadores inválidos
    return -1, -1, -1, -1, -1