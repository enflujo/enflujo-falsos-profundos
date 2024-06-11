# Falsos Profundos

Basado en https://github.com/hacksider/Deep-Live-Cam

## Instalación

- Python (recomendado 3.10)
- FFMPEG
- (en Windows) [Visual Studio 2022 Runtimes](https://visualstudio.microsoft.com/es/visual-cpp-build-tools/)
- [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
- descargar modelo [inswapper_128_fp.onnx](https://huggingface.co/ninjawick/webui-faceswap-unlocked/resolve/main/inswapper_128_fp16.onnx "Modelo") a la carpeta `/modelos`

```bash
pip install -r requirements.txt
```

## Iniciar Programa

```bash
python programa.py --contexto cuda
```

Definiendo recursos:

```bash
python programa.py --contexto cuda --memoria-max 60
```

## Desinstalar todas las dependencias

```bash
pip freeze > unins ; pip uninstall -y -r unins ; del unins
```
