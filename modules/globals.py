import os
from typing import List, Dict

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

source_path = None
target_path = None
output_path = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = []
execution_threads = None
log_level = "error"
