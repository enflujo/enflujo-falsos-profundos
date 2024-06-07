# Falsos Profundos

Basado en https://github.com/hacksider/Deep-Live-Cam

Asume 

## Instalación

- Python (recomendado 3.10)
- FFMPEG
- (en Windows) [Visual Studio 2022 Runtimes](https://visualstudio.microsoft.com/es/visual-cpp-build-tools/)
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)

```bash
pip install -r requirements.txt
```

## Iniciar Programa

```bash
python run.py --execution-provider cuda
```

Definiendo recursos:

```bash
python run.py --execution-provider cuda --execution-threads 60 --max-memory 60
```
