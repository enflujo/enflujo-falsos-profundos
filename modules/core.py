import os
import sys

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

import modules.globals
import modules.ui as ui


warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: quit())
    program = argparse.ArgumentParser()

    program.add_argument(
        "--video-encoder",
        help="adjust output video encoder",
        dest="video_encoder",
        default="libx264",
        choices=["libx264", "libx265", "libvpx-vp9"],
    )

    program.add_argument(
        "--video-quality",
        help="adjust output video quality",
        dest="video_quality",
        type=int,
        default=18,
        choices=range(52),
        metavar="[0-51]",
    )

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

    program.add_argument(
        "--execution-threads",
        help="number of execution threads",
        dest="execution_threads",
        type=int,
        default=suggest_execution_threads(),
    )

    args = program.parse_args()

    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(
        args.execution_provider
    )
    modules.globals.execution_threads = args.execution_threads


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
    if "DmlExecutionProvider" in modules.globals.execution_providers:
        return 1
    if "ROCMExecutionProvider" in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices("GPU")
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024**3
        if platform.system().lower() == "darwin":
            memory = modules.globals.max_memory * 1024**6
        if platform.system().lower() == "windows":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(
                -1, ctypes.c_size_t(memory), ctypes.c_size_t(memory)
            )
        else:
            import resource

            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def run() -> None:
    parse_args()
    limit_resources()
    window = ui.init()
    window.mainloop()
