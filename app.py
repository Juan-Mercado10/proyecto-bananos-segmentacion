"""
Control de calidad de bananos - segmentación por instancia y clasificación de estado.
Aprendizaje Computacional - Universidad de Córdoba.

Flujo: YOLO-World (detecta cada banano) -> MobileSAM (máscara de cada banano)
       -> MobileNetV2 (clasifica el recorte: SANO o DETERIORADO).
Inspección solo visual de la cáscara externa.
"""
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import gradio as gr
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLOWorld, SAM

# En Hugging Face con ZeroGPU hay que marcar la función que usa la GPU.
# En cualquier otro sitio (tu PC, un Space con CPU) este bloque no hace nada.
try:
    import spaces
    gpu = spaces.GPU
except ImportError:
    def gpu(funcion):
        return funcion

PESOS_CLASIFICADOR = 'clasificador_banano_DEFINITIVO.pth'
UMBRAL_INCIERTO = 0.70          # por debajo de esta confianza se marca el banano con "?"
DETECCION_POR_DEFECTO = 0.02    # configuración final de la sección 4 del informe
DETERIORO_POR_DEFECTO = 0.40
NMS_POR_DEFECTO = 0.45

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Clasificador propio (MobileNetV2 con fine-tuning, Ejecución 2) ---
modelo_final = models.mobilenet_v2(weights=None)
modelo_final.classifier[1] = nn.Linear(modelo_final.last_channel, 2)
modelo_final.load_state_dict(torch.load(PESOS_CLASIFICADOR, map_location='cpu'))
modelo_final = modelo_final.to(device).eval()

# --- Modelos preentrenados (se descargan solos la primera vez) ---
detector = YOLOWorld('yolov8s-worldv2.pt')
detector.set_classes(['banana', 'yellow banana', 'brown rotten banana'])
segmentador = SAM('mobile_sam.pt')

normalizar = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def dibujar_etiqueta(img, texto, x, y, color):
    # Escribe el texto con fondo negro y sin salirse de la imagen
    fuente, escala, grosor = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(texto, fuente, escala, grosor)
    x = int(min(max(0, x), max(0, img.shape[1] - tw - 6)))
    y = int(min(max(th + 6, y), img.shape[0] - base - 2))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base), (0, 0, 0), -1)
    cv2.putText(img, texto, (x + 3, y - 3), fuente, escala, color, grosor)


@gpu
def analizar_lote_bananos(imagen_entrada, umbral_deteccion=DETECCION_POR_DEFECTO,
                          umbral_deterioro=DETERIORO_POR_DEFECTO, iou_nms=NMS_POR_DEFECTO):
    if imagen_entrada is None:
        return None, 'Por favor sube una imagen con 3 o más bananos.'

    img_rgb = np.array(imagen_entrada.convert('RGB'))
    alto, ancho, _ = img_rgb.shape
    resultado_visual = img_rgb.copy()

    # 1. Detectar cada banano (iou_nms: cuánto se pueden traslapar dos cajas sin borrar una)
    res_det = detector.predict(img_rgb, conf=umbral_deteccion, iou=iou_nms,
                               agnostic_nms=True, verbose=False)[0]
    if res_det.boxes is None or len(res_det.boxes) == 0:
        return Image.fromarray(resultado_visual), 'No se detectaron bananos. Baja el umbral de detección o revisa la foto.'

    cajas = res_det.boxes.xyxy.cpu().numpy()

    # 2. Máscara de cada banano con MobileSAM
    res_sam = segmentador.predict(img_rgb, bboxes=cajas, verbose=False)[0]
    if res_sam.masks is None:
        return Image.fromarray(resultado_visual), 'No se pudieron generar máscaras.'
    mascaras = res_sam.masks.data.cpu().numpy()

    total, sanos, deteriorados, inciertos = 0, 0, 0, 0
    lista_detalles = []

    for masc_cruda, caja in zip(mascaras, cajas):
        masc_bin = cv2.resize(masc_cruda.astype(np.uint8), (ancho, alto)) > 0

        # Descartar ruido diminuto (< 1.5% del área de la foto)
        if masc_bin.sum() < 0.015 * (alto * ancho):
            continue

        x1, y1, x2, y2 = map(int, caja)
        banano_aislado = img_rgb.copy()
        banano_aislado[~masc_bin] = 0
        recorte = banano_aislado[max(0, y1):min(alto, y2), max(0, x1):min(ancho, x2)]
        if recorte.size == 0:
            continue

        total += 1

        # 3. Clasificar el estado con MobileNetV2
        recorte_pil = Image.fromarray(recorte).resize((224, 224))
        tensor_entrada = normalizar(TF.to_tensor(recorte_pil)).unsqueeze(0).to(device)
        with torch.no_grad():
            prob_deterioro = torch.softmax(modelo_final(tensor_entrada), dim=1)[0, 1].item()

        if prob_deterioro >= umbral_deterioro:
            deteriorados += 1
            estado, confianza, color = 'DETERIORADO', prob_deterioro, (220, 40, 40)
        else:
            sanos += 1
            estado, confianza, color = 'SANO', 1.0 - prob_deterioro, (40, 190, 60)

        incierto = confianza < UMBRAL_INCIERTO
        if incierto:
            inciertos += 1

        overlay_color = np.array(color)
        resultado_visual[masc_bin] = (resultado_visual[masc_bin] * 0.55 + overlay_color * 0.45).astype(np.uint8)
        contornos, _ = cv2.findContours(masc_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(resultado_visual, contornos, -1, color, 2)

        etiqueta = f'ID#{total} {estado} ({confianza:.2f})' + (' ?' if incierto else '')
        ys, xs = np.where(masc_bin)
        dibujar_etiqueta(resultado_visual, etiqueta, int(xs.mean()) - 70, int(ys.mean()), color)
        lista_detalles.append(etiqueta)

    if total == 0:
        return Image.fromarray(resultado_visual), 'Se detectaron objetos, pero ninguno tenía tamaño suficiente.'

    resumen = (
        'RESUMEN DEL LOTE DE BANANOS:\n'
        f'• Total de bananos detectados: {total}\n'
        f'• Bananos sanos: {sanos}\n'
        f'• Bananos deteriorados: {deteriorados}\n'
        f'• Con confianza baja (marcados con ?): {inciertos}\n\n'
        'Detalle por ejemplar:\n' + '\n'.join(lista_detalles)
    )
    return Image.fromarray(resultado_visual), resumen


interfaz = gr.Interface(
    fn=analizar_lote_bananos,
    inputs=[
        gr.Image(type='pil', label='Sube una foto con 3 o más bananos'),
        gr.Slider(0.01, 0.90, value=DETECCION_POR_DEFECTO, step=0.01,
                  label='Umbral de detección (sensibilidad para hallar bananos)'),
        gr.Slider(0.10, 0.90, value=DETERIORO_POR_DEFECTO, step=0.05,
                  label='Umbral de deterioro (sensibilidad para marcar daño)'),
        gr.Slider(0.30, 0.95, value=NMS_POR_DEFECTO, step=0.05,
                  label='Umbral IoU del filtro NMS (más alto = conserva cajas que se traslapan)'),
    ],
    outputs=[
        gr.Image(type='pil', label='Segmentación individual y clasificación'),
        gr.Textbox(label='Conteo y diagnóstico del lote', lines=10),
    ],
    title='Control de Calidad de Banano — Segmentación por instancia',
    description='YOLO-World + MobileSAM (segmentación) y MobileNetV2 (estado: sano o deteriorado). '
                'Inspección solo visual de la cáscara externa.',
)

if __name__ == '__main__':
    interfaz.launch()
