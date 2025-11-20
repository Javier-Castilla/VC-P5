<div align="center">
  <img src="https://www.eii.ulpgc.es/sites/default/files/eii-acron-mod.png"
       alt="Logo ULPGC"
       width="500"
       style="margin-bottom: 10px;">
</div>

<h1 align="center">Práctica 5</h1>

<div align="center" style="font-family: 'Segoe UI', sans-serif; line-height: 1.6; margin-top: 30px;">
  <h2 style="font-size: 28px; margin-bottom: 10px;">
    Asignatura: <span>Visión por Computador</span>
  </h2>
  <p style="font-size: 18px; margin: 4px 0;">
    Grado en Ingeniería Informática
  </p>
  <p style="font-size: 18px; margin-top: 10px;">
    Curso <strong>2025 / 2026</strong>
  </p>
</div>

<h2 align="center">Autores</h2>

- Asmae Ez Zaim Driouch
- Javier Castilla Moreno

<h2 align="center">Bibliotecas utilizadas</h2>

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E?style=for-the-badge&logoColor=black)](https://huggingface.co/docs/transformers)
[![Ultralytics](https://img.shields.io/badge/Ultralytics_YOLO-00A3E0?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Tesseract](https://img.shields.io/badge/Tesseract_OCR-2B579A?style=for-the-badge&logo=python&logoColor=white)](https://github.com/tesseract-ocr/tesseract)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-FF6F61?style=for-the-badge&logo=python&logoColor=white)](https://github.com/JaidedAI/EasyOCR)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org/)
[![Pillow](https://img.shields.io/badge/Pillow-3C8D40?style=for-the-badge&logo=python&logoColor=white)](https://pillow.readthedocs.io/)

## Cómo usar
### Primer paso: clonar este repositorio
```bash
git clone "https://github.com/A-NullPointer/VC-P4-P4b"
```
### Segundo paso: Activar tu environment e instalar dependencias
> [!NOTE]
> Todas las dependencias pueden verse en [este archivo](environment.yml). Si se desea, puede crearse un entorno de Conda con dicho archivo.

Este environment es si cuenta con **GPU**. Por ello necesita tener instalado CUDA. Para ello puede ejcutar el siguiente comando:

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Ahora sí se procede a la creación del entorno empleando el environment a partir del archivo expuesto, es necesario abrir el `Anaconda Prompt` y ejecutar lo siguiente:

```bash
conda env create -f environment.yml
```

Posteriormente, se activa el entorno:

```bash
conda activate VC_P4
```


### Tercer paso: ejecutar el cuaderno
Finalmente, abriendo nuestro IDE favorito y teniendo instalado todo lo necesario para poder ejecutar notebooks, se puede ejecutar los cuadernos de la práctica [Practica_4.ipynb](P4/practica_4.ipynb) y [Practica_4b.ipynb](P4/practica_4b.ipynb) seleccionando el environment anteriormente creado.

> [!IMPORTANT]
> Todos los bloques de código deben ejecutarse en orden, de lo contrario, podría ocasionar problemas durante la ejecución del cuaderno.

<h1 align="center">Tareas Práctica 5</h1>

<h2 align="center">Prototipo 1: Detector de Emociones</h2>

<h2 align="center">Prototipo 2: Métricas de Atención</h2>

<h1 align="center">Bibliografía</h1>
