import os
import customtkinter as ctk
from typing import Tuple
import cv2
from PIL import Image, ImageOps
from modules.predicter import predict_frame

import modules.globals
from modules.face_analyser import get_one_face
from modules.capturer import get_video_frame, get_video_frame_total
from modules.utilities import is_image, is_video, resolve_relative_path
from modules.processors.frame import face_swapper

ROOT = None
ROOT_HEIGHT = 700
ROOT_WIDTH = 600

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

preview_label = None
preview_slider = None
source_label = None

img_ft, vid_ft = modules.globals.file_types


def init() -> ctk.CTk:
    global ROOT, PREVIEW

    ROOT = create_root()
    PREVIEW = create_preview(ROOT)

    return ROOT


def create_root() -> ctk.CTk:
    global source_label

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.title(f"EnFlujo")
    root.configure()
    # root.protocol("WM_DELETE_WINDOW", lambda: exit())

    source_label = ctk.CTkLabel(root, text="")
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    source_button = ctk.CTkButton(
        root,
        text="Foto",
        cursor="hand2",
        command=lambda: select_source_path(),
    )
    source_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    return root


def create_preview(parent: ctk.CTk) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title("Espejito Espejito")
    preview.configure()
    preview.resizable(width=False, height=False)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()

    source_path = ctk.filedialog.askopenfilename(
        title="select an source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image)
        webcam_preview()
    else:
        modules.globals.source_path = None
        source_label.configure(image=None)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
    video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()

    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()

    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)

        if predict_frame(temp_frame):
            quit()
        temp_frame = face_swapper.process_frame(
            get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
        )
        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)


def webcam_preview():
    if modules.globals.source_path is None:
        # No image selected
        return

    global preview_label, PREVIEW

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)  # Set the width of the resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # Set the height of the resolution
    cap.set(cv2.CAP_PROP_FPS, 60)  # Set the frame rate of the webcam
    PREVIEW_MAX_WIDTH = 960
    PREVIEW_MAX_HEIGHT = 540

    preview_label.configure(image=None)  # Reset the preview image before startup
    PREVIEW.deiconify()  # Open preview window

    source_image = None  # Initialize variable for the selected face image

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Select and save face image only once
        if source_image is None and modules.globals.source_path:
            source_image = get_one_face(cv2.imread(modules.globals.source_path))

        temp_frame = frame.copy()  # Create a copy of the frame
        temp_frame = face_swapper.process_frame(source_image, temp_frame)

        image = cv2.cvtColor(
            temp_frame, cv2.COLOR_BGR2RGB
        )  # Convert the image to RGB format to display it with Tkinter
        image = Image.fromarray(image)
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        ROOT.update()

    cap.release()
    PREVIEW.withdraw()  # Close preview window when loop is finished
