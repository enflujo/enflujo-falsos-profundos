from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.globals


def multi_process_frame(
    source_path: str,
    temp_frame_paths: List[str],
    process_frames: Callable[[str, List[str], Any], None],
    progress: Any = None,
) -> None:
    with ThreadPoolExecutor(max_workers=modules.globals.execution_threads) as executor:
        futures = []
        for path in temp_frame_paths:
            future = executor.submit(process_frames, source_path, [path], progress)
            futures.append(future)
        for future in futures:
            future.result()


def process_video(
    source_path: str,
    frame_paths: list[str],
    process_frames: Callable[[str, List[str], Any], None],
) -> None:
    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(frame_paths)
    with tqdm(
        total=total,
        desc="Processing",
        unit="frame",
        dynamic_ncols=True,
        bar_format=progress_bar_format,
    ) as progress:
        progress.set_postfix(
            {
                "execution_providers": modules.globals.execution_providers,
                "execution_threads": modules.globals.execution_threads,
                "max_memory": modules.globals.max_memory,
            }
        )
        multi_process_frame(source_path, frame_paths, process_frames, progress)
