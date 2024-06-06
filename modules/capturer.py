import cv2
import cv2.typing
from typing import Optional


def get_video_frame(
    video_path: str, frame_number: int = 0
) -> Optional[tuple[bool, cv2.typing.MatLike]]:
    capture = cv2.VideoCapture(video_path)
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
    frame = capture.read()
    capture.release()

    if frame:
        return frame
    return None


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
