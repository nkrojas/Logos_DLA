---
title: Generador Logos
emoji: 🐠
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.0.2
app_file: app.py
pinned: false
license: mit
short_description: Generador de logos para emprendimientos
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


--- 
GANs para Generación de Logos
---

Descripción General:


Este proyecto implementa y compara tres arquitecturas de Redes Generativas Adversariales (GANs) basadas en el modelo DCGAN para la tarea de generación de imágenes de logos. Los experimentos evalúan el impacto de diferentes técnicas de normalización y regularización en la estabilidad del entrenamiento y en la calidad de las imágenes generadas, medidas a través de las métricas FID e Inception Score.

---
Experimentos:
---

Exp1 (Base): Configuración estándar de DCGAN.

Exp2 (SpectralNorm): Incorpora Normalización Espectral en el Discriminador.

Exp3 (Advanced): Incluye técnicas avanzadas de optimización y arquitectura.

---
Requisitos Previos:
---

Asegúrate de tener instalado lo siguiente antes de comenzar:

Git: Para clonar el repositorio.

Python 3.8+: Es el entorno recomendado.

Gestor de Entornos: Se recomienda usar venv o conda.

---
Dependencias de Hardware
---

El entrenamiento es intensivo en cómputo. Se recomienda encarecidamente una:

GPU NVIDIA: Con soporte para CUDA (versión 11.x o superior) para acelerar el entrenamiento a través de PyTorch (Opciones de Ejecución COLAB)

---
Instalación:
---

Sigue estos pasos para configurar el entorno de trabajo:

1. Clonar el Repositorio
   git clone https://github.com/nkrojas/Logos_DLA.git
   cd Logos_DLA


2. Crear y Activar Entorno Virtual
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   .\venv\Scripts\activate  # Windows
    
3. Instalar Bibliotecas Requeridas
   Este proyecto depende del framework PyTorch y librerías auxiliares como torch_fidelity. Instala las dependencias:
   pip install -r requirements.txt

---
Uso y Ejecución:
---

1. Preparación de Datos

Ubicación: El dataset de imágenes de logos debe colocarse en un directorio accesible.
Configuración: La ruta del dataset se especifica mediante el argumento --dataroot en el script de entrenamiento.
Ejemplo: Si las imágenes están en ./data/logos/, el comando usará --dataroot ./data/logos/.

2. Ejecutar el Entrenamiento

El script principal es train.py. Los resultados (modelos guardados, imágenes generadas) se almacenarán en el directorio ./results/.
Para ejecutar cada experimento individualmente:

| Experimento | Comando de Ejecución | Descripción|
| --- | --- | --- |
| Exp1 | (Base),python train.py --experiment exp1 --n_epochs 100 --batch_size 64, | Entrenamiento DCGAN estándar. |
| Exp2 | (SpectralNorm),python train.py --experiment exp2 --n_epochs 100 --batch_size 64 --spectral_norm true | Aplica Normalización Espectral. |
| Exp3 | (Advanced),python train.py --experiment exp3 --n_epochs 100 --batch_size 64 --advanced true | Utiliza la configuración avanzada. |


(Nota: Ajusta los valores de --n_epochs y --batch_size según tus recursos de hardware.)

---
Evaluación de Resultados:
---

Una vez finalizado el entrenamiento, puedes evaluar la calidad de los modelos generados.

Gráficas de Pérdida: Las curvas de Loss_D y Loss_G (comparativas como las mostradas) se guardarán automáticamente en el directorio de resultados.

Métricas de Calidad (FID / IS): Para calcular la distancia entre las imágenes reales y las generadas.
