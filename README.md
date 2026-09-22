# README del proyecto de visión artificial

Este repositorio reúne una colección de ejercicios y pruebas de visión por computador, detección de objetos, reconstrucción 3D, realidad aumentada y deep learning. La mayor parte de la lógica está en la carpeta `code/`, mientras que la documentación de referencia se guarda en `doc/`.

La estructura principal es la siguiente:

```text
.
├── README.md
├── code/
│   ├── calibrate/
│   ├── clasificador/
│   ├── detector/
│   ├── rectificacion/
│   ├── trafico/
│   ├── opcional-colmap/
│   ├── opcional-lk/
│   ├── opcional-ra/
│   ├── opcional-UMU/
│   ├── DL/
│   └── ...
└── doc/
    └── VIA_mem.pdf
```

---

## 1. Resumen general

Los ejercicios están orientados a:

- calibración de cámaras y medición geométrica,
- clasificación visual por descriptors y embeddings,
- detección y seguimiento de objetos,
- conteo y análisis de tráfico,
- rectificación de perspectiva y homografía,
- reconstrucción 3D con SfM,
- flujo óptico y movimiento de cámara,
- realidad aumentada con marcadores,
- segmentación semántica con redes U-Net.

La explicación más detallada de algunos ejercicios está en notebooks Jupyter dentro de sus carpetas. En esos casos, el script Python aporta la implementación interactiva, pero la teoría y la guía están en el notebook.

---

## 2. Ejercicios principales de la carpeta `code/`

### 2.1. `code/calibrate/`

Ejercicio de calibración y medida de distancia geométrica en imagen.

- Archivo principal: `code/calibrate/cuadricula.py`
- Objetivo:
  - leer una matriz de calibración desde `calib.txt`,
  - proyectar una cuadrícula virtual en la escena,
  - estimar distancias reales a partir de un punto de referencia,
  - visualizar una malla de referencia en el plano con medidas en centímetros.
- Idea del ejercicio:
  - se usa modelado de cámara pinhole,
  - se dibujan rejillas en profundidad y altura,
  - el usuario puede pinchar dos puntos para medir una distancia.

Uso típico:

```bash
cd code/calibrate
python cuadricula.py
```

---

### 2.2. `code/clasificador/`

Ejercicio de clasificación visual por comparación con modelos guardados.

- Archivo principal: `code/clasificador/clasificador.py`
- Carpetas importantes:
  - `code/clasificador/methods/`
  - `code/clasificador/examples/`
  - `code/clasificador/models/`
- Métodos disponibles:
  - `sift.py` → comparación por SIFT
  - `hand.py` → descriptors manuales
  - `embedder.py` → embedding con modelo TensorFlow Lite
- Objetivo:
  - cargar modelos de referencia,
  - extraer descriptores del frame actual,
  - comparar con modelos almacenados,
  - guardar nuevas capturas en la carpeta de modelos.

Uso típico:

```bash
cd code/clasificador
python clasificador.py --models models --method sift
```

Este ejercicio sirve para comparar múltiples estrategias de reconocimiento visual sin cambiar la interfaz principal del programa.

---

### 2.3. `code/detector/`

Ejercicio de detección de movimiento y personas con YOLO + seguimiento de vídeo.

- Archivo principal: `code/detector/detector.py`
- Objetivo:
  - definir una región de interés (ROI),
  - detectar movimiento con MOG2,
  - detectar personas con YOLO (`yolo11n.pt`),
  - anonimizar caras detectadas,
  - grabar un vídeo cuando hay actividad y enviar por Telegram si se confirma presencia humana.
- Características:
  - uso de `ultralytics`,
  - contador de movimiento y umbrales de confianza,
  - grabación automática de secuencias cortas en formato video.

Uso típico:

```bash
cd code/detector
python detector.py
```

---

### 2.4. `code/rectificacion/`

Ejercicio de homografía y rectificación de una vista plana.

- Archivo principal: `code/rectificacion/rectificacion.py`
- Objetivo:
  - cargar un archivo de correspondencias 2D-3D (`ref-*.txt`),
  - estimar una homografía,
  - rectificar la imagen para verla desde un plano de referencia,
  - realizar mediciones reales en unidades definidas por el usuario.
- Archivos de referencia:
  - `ref-coins.txt`
  - `ref-objetos.txt`

Uso típico:

```bash
cd code/rectificacion
python rectificacion.py --ref ref-objetos.txt --units cm
```

También admite el modo interactivo para crear nuevos puntos de referencia con `--pick`.

---

### 2.5. `code/trafico/`

Ejercicio de conteo de tráfico y seguimiento vehicular.

- Archivo principal: `code/trafico/trafico.py`
- Objetivo:
  - aplicar substracción de fondo,
  - detectar contornos de vehículos,
  - seguir centros de masa,
  - contar coches que cruzan una línea central,
  - distinguir dirección izquierda-derecha y derecha-izquierda.
- Se usa una ROI para limitar la zona de análisis.

Uso típico:

```bash
cd code/trafico
python trafico.py
```

---

## 3. Ejercicios opcionales

### 3.1. `code/opcional-lk/`

Ejercicio de flujo óptico de Lucas-Kanade.

- Archivo principal: `code/opcional-lk/lk_motion.py`
- Objetivo:
  - detectar puntos de interés,
  - estimar movimiento entre frames,
  - clasificar dirección dominante: `UP`, `DOWN`, `LEFT`, `RIGHT`, `FORWARD`, `BACKWARD`, `STATIC`,
  - estimar velocidad angular de la cámara.
- Este ejercicio se centra en movimiento de cámara y análisis de flujo óptico.

---

### 3.2. `code/opcional-ra/`

Ejercicio de realidad aumentada con marcador planar.

- Archivos relevantes:
  - `code/opcional-ra/ar_mouse.py`
  - `code/opcional-ra/ar_mouse.ipynb`
- Objetivo:
  - detectar un marcador planar en la imagen,
  - estimar su pose 3D,
  - proyectar una esfera virtual sobre el plano,
  - permitir interactuar con el ratón para desplazar la esfera sobre el marcador.
- La explicación teórica y visual completa está en el notebook `ar_mouse.ipynb`.

---

### 3.3. `code/opcional-UMU/`

Ejercicio de detección y reprojectación de un carnet o tarjeta sobre imagen.

- Archivo principal: `code/opcional-UMU/card.py`
- Archivo notebook: `code/opcional-UMU/ejer-CARD.ipynb`
- Objetivo:
  - detectar un documento o tarjeta en la imagen,
  - estimar sus esquinas,
  - corregir la perspectiva,
  - superponer la foto del carnet o una imagen sobre el rectángulo.

---

### 3.4. `code/opcional-colmap/`

Ejercicio de reconstrucción 3D mediante Structure-from-Motion.

- Archivo notebook: `code/opcional-colmap/colmap.ipynb`
- Objetivo:
  - extraer frames de un video,
  - aplicar COLMAP para reconstruir un modelo 3D,
  - comparar la salida clásica de COLMAP con la de `vggt` (modelo basado en transformers).
- En esta carpeta se incluyen imágenes y salida de ejemplo del proceso.

---

## 4. Deep Learning (`code/DL/`)

La carpeta de deep learning contiene un laboratorio de segmentación semántica con U-Net.

- Notebook principal: `code/DL/deepLearning.ipynb`
- Directorios relevantes:
  - `code/DL/caras_mias/`
  - `code/DL/caras_mias_test/`
  - `code/DL/caras.torch`
- Objetivo:
  - crear máscaras manuales sobre caras,
  - preparar dataset con recortes y aumento de datos,
  - entrenar una U-Net para segmentar rostros,
  - validar la inferencia sobre imágenes reales.

El notebook documenta el flujo completo junto con la arquitectura del modelo y la preparación del dataset.

---

## 5. Documentación adicional

La carpeta `doc/` contiene el documento de referencia:

- [doc/VIA_mem.pdf](doc/VIA_mem.pdf)

Este PDF funciona como material de apoyo del curso o de la asignatura, pero la ejecución y la explicación práctica de muchos ejercicios está distribuida entre scripts Python y notebooks.

---

## 6. Recomendación de uso

Si quieres seguir el proyecto de forma ordenada, se recomienda este flujo:

1. empezar por los ejercicios básicos:
   - `calibrate/`
   - `rectificacion/`
   - `trafico/`
2. continuar con la parte de detección y clasificación:
   - `detector/`
   - `clasificador/`
3. abrir los notebooks de los ejercicios opcionales:
   - `opcional-colmap/colmap.ipynb`
   - `opcional-ra/ar_mouse.ipynb`
   - `opcional-UMU/ejer-CARD.ipynb`
   - `DL/deepLearning.ipynb`

---

## 7. Notas finales

- Algunos ejercicios tienen explicación principal en notebook, no solo en el script.
- Los scripts están diseñados para ejecutarse en entorno con OpenCV, NumPy y, en algunos casos, librerías adicionales como `ultralytics` o `torch`.
- Si quieres más detalle técnico sobre un ejercicio concreto, abre primero el notebook correspondiente y luego el script Python asociado.

---

## 8. Inicio rápido

```bash
# Ejemplo de ejecución del ejercicio de calibración
cd code/calibrate
python cuadricula.py

# Ejemplo de ejecución del clasificador
cd ../clasificador
python clasificador.py --models models --method sift

# Ejemplo de ejecución de rectificación
cd ../rectificacion
python rectificacion.py --ref ref-objetos.txt --units cm
```
