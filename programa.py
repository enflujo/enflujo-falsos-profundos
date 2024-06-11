import os

# single thread doubles cuda performance - needs to be set before torch import
os.environ["OMP_NUM_THREADS"] = "1"
# reduce tensorflow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import torch  # se tiene que importar sin importar que no se ejecute
import warnings
import signal
from argparse import ArgumentParser
import cv2
from utilidades.ayudas import (
    sugerenciaMemoriaRam,
    codificarProveedores,
    decodificarProveedores,
    limitarRecursos,
    esImagen,
    resolverRuta,
)

import customtkinter as ctk
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.app.common import Face
from insightface.model_zoo.inswapper import INSwapper
from PIL import Image, ImageOps

# Tipos
from cv2.typing import MatLike
from customtkinter import CTk, CTkLabel, CTkToplevel
from typing import Tuple, Optional, List, cast

dimsLienzo = (1200, 700)
mostrarMuchasCaras = True


warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
interfaz: CTk
lienzo: CTkLabel
ventanaCamara: CTkToplevel
analizador: FaceAnalysis
modelo: INSwapper


def configurarPrograma() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: quit())
    programa = ArgumentParser()

    programa.add_argument(
        "--max-memory",
        help="máximo de RAM en GB",
        dest="memoriaMax",
        type=int,
        default=sugerenciaMemoriaRam(),
    )

    programa.add_argument(
        "--contexto",
        help="Puede ser cpu o cuda",
        dest="contexto",
        default=["cpu"],
        choices=codificarProveedores(),
        nargs="+",
    )

    args = programa.parse_args()

    memoriaMaxima = args.memoriaMax
    proveedores = decodificarProveedores(args.contexto)

    limitarRecursos(memoriaMaxima)
    cargarModelos(proveedores)


def crearInterfaz() -> None:
    global interfaz, lienzo, ventanaCamara
    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("dark-blue")

    interfaz = CTk()
    interfaz.minsize(600, 700)
    interfaz.title("..:: Falsos Profundos | EnFlujo ::..")
    interfaz.wm_iconbitmap("favicon.ico")
    interfaz.configure()

    ventanaImagen = CTkLabel(interfaz, text="")
    ventanaImagen.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.25)

    ventanaCamara = CTkToplevel(interfaz)
    ventanaCamara.withdraw()
    ventanaCamara.title("Espejito Espejito")
    ventanaCamara.wm_iconbitmap("favicon.ico")
    ventanaCamara.configure()
    ventanaCamara.resizable(width=False, height=False)

    lienzo = CTkLabel(ventanaCamara, text="")
    lienzo.pack(fill="both", expand=True)

    botonImagen = ctk.CTkButton(
        interfaz,
        text="Foto",
        cursor="hand2",
        command=lambda: seleccionarImagen(ventanaImagen),
    )
    botonImagen.place(relx=0.1, rely=0.4, relwidth=0.3, relheight=0.1)

    interfaz.mainloop()


def abrirImagen(ruta: str, tamaño: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(ruta)
    if tamaño:
        image = ImageOps.fit(image, tamaño, Image.Resampling.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def seleccionarImagen(ventanaImagen: CTkLabel) -> None:
    ventanaCamara.withdraw()

    rutaImagen = ctk.filedialog.askopenfilename(
        title="Seleccionar una imagen de donde extraer la cara",
        initialdir=None,
        filetypes=(("Imágen", ("*.png", "*.jpg", "*.gif", "*.bmp")),),
    )

    if esImagen(rutaImagen):
        image = abrirImagen(rutaImagen, (200, 200))
        ventanaImagen.configure(image=image)
        iniciarCamara(rutaImagen)
    else:
        ventanaImagen.configure(image=None)


def cargarModelos(proveedores: List[str]) -> None:
    global modelo, analizador
    analizador = FaceAnalysis(name="buffalo_l", providers=proveedores)
    analizador.prepare(ctx_id=0, det_size=(640, 640))
    rutaModelo = resolverRuta("../modelos/inswapper_128_fp16.onnx")
    modelo = cast(INSwapper, get_model(rutaModelo, providers=proveedores))


def unaCara(fotograma: MatLike) -> Optional[Face]:
    cara = analizador.get(fotograma)
    try:
        res = min(cara, key=lambda x: x.bbox[0])
        return res
    except ValueError:
        return None


def muchasCaras(fotograma: MatLike) -> Optional[list[Face]]:
    try:
        return analizador.get(fotograma)
    except IndexError:
        return None


def falsoProfundo(caraOriginal: Face, caraCamara: Face, fotograma: MatLike) -> MatLike:
    return modelo.get(fotograma, caraCamara, caraOriginal, paste_back=True)  # type: ignore


def procesar(caraOriginal: Face, fotograma: MatLike) -> MatLike:
    if mostrarMuchasCaras:
        caras = muchasCaras(fotograma)
        if caras:
            for cara in caras:
                fotograma = falsoProfundo(caraOriginal, cara, fotograma)
    else:
        cara = unaCara(fotograma)
        if cara:
            fotograma = falsoProfundo(caraOriginal, cara, fotograma)
    return fotograma


def iniciarCamara(rutaImagen: str):
    caraOriginal = unaCara(cv2.imread(rutaImagen))
    if caraOriginal is None:
        return

    # Iniciar cámara
    camara = cv2.VideoCapture(0)
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    camara.set(cv2.CAP_PROP_FPS, 60)

    lienzo.configure(image=None)
    ventanaCamara.deiconify()  # Abrir ventana de video

    while True:
        ret, frame = camara.read()

        if not ret:
            break

        fotograma = cv2.flip(frame, 1)  # Invertir horizontalmente
        copia = fotograma.copy()  # Copia de la imagen
        copia = procesar(caraOriginal, copia)  # extraer y reemplazar caras

        # Tkinter necesita la imagen en formato RGB
        imagen = cv2.cvtColor(copia, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(imagen)

        imagen = ImageOps.contain(
            imagen, (dimsLienzo[0], dimsLienzo[1]), Image.Resampling.LANCZOS
        )
        imagen = ctk.CTkImage(imagen, size=imagen.size)
        lienzo.configure(image=imagen)
        interfaz.update()

    camara.release()
    ventanaCamara.withdraw()  # Cerrar ventana de video cuando termine o se apague el programa


configurarPrograma()
crearInterfaz()
