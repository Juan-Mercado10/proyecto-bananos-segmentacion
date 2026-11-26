# Control de calidad de bananos — segmentación por instancia y clasificación de estado

Proyecto final de **Aprendizaje Computacional** (Ingeniería de Sistemas, Universidad de Córdoba).
Integrantes: Juan Diego Mercado, Aizar Racero y Diego Caro. Docente: Oswaldo Vélez Langs.

A partir de una foto con varios bananos, la app delimita cada banano con su propia máscara,
le asigna un ID, lo clasifica como **SANO** o **DETERIORADO** con su confianza (con un `?` si
es menor a 0.70) y muestra el resumen: total, sanos y deteriorados.
La inspección es solo visual (cáscara externa): no certifica inocuidad ni detecta daños internos.

**Demo en línea:** \[COMPLETAR enlace del Space]

## Cómo funciona

1. **YOLO-World** (`yolov8s-worldv2`) encuentra cada banano (cajas).
2. **MobileSAM** saca la máscara de cada banano.
3. **MobileNetV2** (pesos ImageNet + fine-tuning propio) clasifica el recorte: sano o deteriorado.
Si la probabilidad de deterioro es ≥ 0.40, se marca DETERIORADO.

Valores por defecto (los de la configuración final del informe): detección 0.02, deterioro 0.40, NMS 0.45.

## Archivos

|Archivo|Qué es|
|-|-|
|`app.py`|Código de la app (interfaz Gradio)|
|`requirements.txt`|Dependencias|
|`clasificador\_banano\_DEFINITIVO.pth`|Pesos del clasificador final (Ejecución 2, partición definitiva de 4027 imágenes, semilla 42)|

## Instalación y ejecución

Necesita Python 3.10 o superior.

```bash
pip install -r requirements.txt
python app.py
```

Abre en el navegador la dirección que aparece en la terminal (normalmente `http://127.0.0.1:7860`).
La primera vez se descargan solos los pesos de YOLO-World y MobileSAM. Sin GPU funciona, pero es más lento.

## Procedencia de datos y componentes

* **Datos:** dataset público *Fruits fresh and rotten for classification* (Kaggle, usuario `sriramr`),
solo las carpetas de banano (`freshbanana`, `rottenbanana`). Licencia desconocida: uso solo educativo.
Se limpiaron imágenes repetidas (4716 → 4027) y se dividió 70/15/15 antes del aumento de datos.
* **Componentes externos:** Ultralytics YOLOv8s-Worldv2 y MobileSAM (AGPL-3.0), Torchvision MobileNetV2
(pesos ImageNet, BSD-3) y Gradio (Apache 2.0).
* **Propio:** limpieza y partición de datos, preparación, entrenamiento y evaluación del clasificador,
umbrales, evaluación con fotos propias y la interfaz.

## Límites

Funciona mejor con bananos sobre una mesa, separados o poco encimados. En montones apretados puede fusionar
varios bananos en una sola máscara, y tiende a marcar como deteriorados algunos bananos sanos.
El clasificador no tiene la clase "no es banano". Ver el informe, secciones 3 y 4.

