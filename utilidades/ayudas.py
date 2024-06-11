import mimetypes
import os
import platform
import ssl
import onnxruntime
import tensorflow
from typing import List

# monkey patch ssl for mac
if platform.system().lower() == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


def esImagen(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith("image/"))
    return False


def resolverRuta(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def sugerenciaMemoriaRam() -> int:
    if platform.system().lower() == "darwin":
        return 4
    return 16


def codificarProveedores() -> List[str]:
    proveedores: List[str] = onnxruntime.get_available_providers()
    return [
        proveedor.replace("ExecutionProvider", "").lower() for proveedor in proveedores
    ]


def decodificarProveedores(proveedores: List[str]) -> List[str]:
    return [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            codificarProveedores(),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in proveedores
        )
    ]


def limitarRecursos(memoriaMax: int) -> None:
    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)

    memoria = memoriaMax * 1024**3

    if platform.system().lower() == "darwin":
        memoria = memoriaMax * 1024**6

    if platform.system().lower() == "windows":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetProcessWorkingSetSize(
            -1, ctypes.c_size_t(memoria), ctypes.c_size_t(memoria)
        )
    else:
        import resource

        resource.setrlimit(resource.RLIMIT_DATA, (memoria, memoria))  # type: ignore
