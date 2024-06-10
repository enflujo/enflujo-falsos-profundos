import customtkinter as ctk

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper

import cv2
import modulos.globals
from modulos.ayudas import is_image, resolve_relative_path

# Tipos
from cv2.typing import MatLike
from typing import Tuple, Optional, cast
from PIL import Image, ImageOps


PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200

RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

MAX_PROBABILITY = 0.85
MUCHAS_CARAS = True


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
    preview.wm_iconbitmap("favicon.ico")
    preview.configure()
    preview.resizable(width=False, height=False)
    preview_label = ctk.CTkLabel(preview, text="")
    preview_label.pack(fill="both", expand=True)

    analizador = FaceAnalysis(
        name="buffalo_l", providers=modulos.globals.execution_providers
    )
    analizador.prepare(ctx_id=0, det_size=(640, 640))

    rutaModelo = resolve_relative_path("../modelos/inswapper_128_fp16.onnx")
    modelo: INSwapper = cast(
        INSwapper, get_model(rutaModelo, providers=modulos.globals.execution_providers)
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
        return modelo.get(temp_frame, target_face, source_face, paste_back=True)  # type: ignore

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
        # Cargar imagen fuente del falso profundo
        if modulos.globals.source_path is None:
            return
        source_image = get_one_face(cv2.imread(modulos.globals.source_path))
        if source_image is None:
            return

        # Iniciar cámara
        camara = cv2.VideoCapture(0)
        camara.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        camara.set(cv2.CAP_PROP_FPS, 60)

        preview_label.configure(image=None)
        preview.deiconify()  # Abrir ventana de video

        while True:
            ret, frame = camara.read()

            if not ret:
                break
            frame = cv2.flip(frame, 1)
            copia = frame.copy()  # Copia de la imagen
            copia = process_frame(source_image, copia)

            image = cv2.cvtColor(
                copia, cv2.COLOR_BGR2RGB
            )  # Convert the image to RGB format to display it with Tkinter
            image = Image.fromarray(image)

            image = ImageOps.contain(
                image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.Resampling.LANCZOS
            )
            image = ctk.CTkImage(image, size=image.size)
            preview_label.configure(image=image)
            interfaz.update()

        camara.release()
        preview.withdraw()  # Cerrar ventana de video cuando termine o se apague el programa

    def select_source_path() -> None:
        preview.withdraw()

        source_path = ctk.filedialog.askopenfilename(
            title="Seleccionar una imagen de donde extraer la cara",
            initialdir=None,
            filetypes=(("Imágen", ("*.png", "*.jpg", "*.gif", "*.bmp")),),
        )

        if is_image(source_path):
            modulos.globals.source_path = source_path
            image = render_image_preview(modulos.globals.source_path, (200, 200))
            source_label.configure(image=image)

            webcam_preview()
        else:
            modulos.globals.source_path = None
            source_label.configure(image=None)

    def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
        image = Image.open(image_path)
        if size:
            image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        return ctk.CTkImage(image, size=image.size)

    return interfaz
