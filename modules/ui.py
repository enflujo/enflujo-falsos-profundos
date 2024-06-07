import customtkinter as ctk

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

import cv2
from cv2.typing import MatLike

from typing import Tuple, Optional
from PIL import Image, ImageOps

import modules.globals
from modules.utilities import is_image, resolve_relative_path

PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

MAX_PROBABILITY = 0.85
MUCHAS_CARAS = False

img_ft, vid_ft = modules.globals.file_types


def init() -> ctk.CTk:
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    interfaz = ctk.CTk()
    interfaz.minsize(600, 700)
    interfaz.title("..:: Falsos Profundos | EnFlujo ::..")
    interfaz.wm_iconbitmap("favicon.ico")
    interfaz.configure()

    source_label = ctk.CTkLabel(interfaz, text="")
    source_label.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    source_button = ctk.CTkButton(
        interfaz,
        text="Foto",
        cursor="hand2",
        command=lambda: select_source_path(),
    )
    source_button.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    preview = ctk.CTkToplevel(interfaz)
    preview.withdraw()
    preview.title("Espejito Espejito")
    preview.configure()
    preview.resizable(width=False, height=False)
    preview_label = ctk.CTkLabel(preview, text="")
    preview_label.pack(fill="both", expand=True)

    analizador = FaceAnalysis(
        name="buffalo_l", providers=modules.globals.execution_providers
    )
    analizador.prepare(ctx_id=0, det_size=(640, 640))

    rutaModelo = resolve_relative_path("../models/inswapper_128_fp16.onnx")
    modelo: INSwapper = get_model(
        rutaModelo, providers=modules.globals.execution_providers
    )

    def get_one_face(fotograma: MatLike) -> Optional[Face]:
        face = analizador.get(fotograma)
        try:
            res = min(face, key=lambda x: x.bbox[0])
            return res
        except ValueError:
            return None

    def get_many_faces(fotograma: MatLike) -> Optional[list[Face]]:
        try:
            return analizador.get(fotograma)
        except IndexError:
            return None

    def swap_face(source_face: Face, target_face: Face, temp_frame: MatLike) -> MatLike:
        return modelo.get(temp_frame, target_face, source_face, paste_back=True)

    def process_frame(source_face: Face, temp_frame: MatLike) -> MatLike:
        if MUCHAS_CARAS:
            many_faces = get_many_faces(temp_frame)
            if many_faces:
                for target_face in many_faces:
                    temp_frame = swap_face(source_face, target_face, temp_frame)
        else:
            target_face = get_one_face(temp_frame)
            if target_face:
                temp_frame = swap_face(source_face, target_face, temp_frame)
        return temp_frame

    def webcam_preview():
        if modules.globals.source_path is None:
            # No image selected
            return
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        cap.set(cv2.CAP_PROP_FPS, 60)

        preview_label.configure(image=None)  # Reset the preview image before startup
        preview.deiconify()  # Open preview window

        source_image = None  # Initialize variable for the selected face image

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Select and save face image only once
            if source_image is None and modules.globals.source_path:
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            temp_frame = frame.copy()  # Create a copy of the frame
            temp_frame = process_frame(source_image, temp_frame)

            image = cv2.cvtColor(
                temp_frame, cv2.COLOR_BGR2RGB
            )  # Convert the image to RGB format to display it with Tkinter
            image = Image.fromarray(image)

            image = ImageOps.contain(
                image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            image = ctk.CTkImage(image, size=image.size)
            preview_label.configure(image=image)
            interfaz.update()

        cap.release()
        preview.withdraw()  # Close preview window when loop is finished

    def select_source_path() -> None:
        global img_ft, vid_ft

        preview.withdraw()

        source_path = ctk.filedialog.askopenfilename(
            title="Seleccionar una imagen de donde extraer la cara",
            initialdir=None,
            filetypes=[img_ft],
        )

        if is_image(source_path):
            modules.globals.source_path = source_path
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

    return interfaz
