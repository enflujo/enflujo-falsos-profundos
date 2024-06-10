import os

# single thread doubles cuda performance - needs to be set before torch import
os.environ["OMP_NUM_THREADS"] = "1"
# reduce tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import warnings
from typing import List
import platform
import signal
import argparse
import torch  # se tiene que importar sin importar que no se ejecute
import onnxruntime
import tensorflow

import modulos.globals
import modulos.ui as ui


warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: quit())
    program = argparse.ArgumentParser()

    program.add_argument(
        "--max-memory",
        help="maximum amount of RAM in GB",
        dest="max_memory",
        type=int,
        default=suggest_max_memory(),
    )

    program.add_argument(
        "--execution-provider",
        help="execution provider",
        dest="execution_provider",
        default=["cpu"],
        choices=encode_execution_providers(onnxruntime.get_available_providers()),
        nargs="+",
    )

    args = program.parse_args()

    modulos.globals.max_memory = args.max_memory
    modulos.globals.execution_providers = decode_execution_providers(
        args.execution_provider
    )


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers()),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]


def suggest_max_memory() -> int:
    if platform.system().lower() == "darwin":
        return 4
    return 16


def suggest_execution_threads() -> int:
    if "DmlExecutionProvider" in modulos.globals.execution_providers:
        return 1
    if "ROCMExecutionProvider" in modulos.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modulos.globals.max_memory:
        memory = modulos.globals.max_memory * 1024**3
        if platform.system().lower() == "darwin":
            memory = modulos.globals.max_memory * 1024**6
        if platform.system().lower() == "windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(memory), ctypes.c_size_t(memory)
            )
        else:
            import resource

            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))  # type: ignore


parse_args()
limit_resources()
window = ui.init()
window.mainloop()
