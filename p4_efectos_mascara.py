from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

originales = ["fig_01.jpg", "fig_02.jpg", "fig_03.jpg", "fig_04.jpg"]
plantillas = ["pla_01.jpg", "pla_02.jpg", "pla_03.jpg", "pla_04.jpg"]

def efecto(fondo, plantilla, lena="imagenes/fig_00.jpg"):
    fondo = Image.open(fondo).convert("RGB")
    lena = Image.open(lena).convert("RGB").resize(fondo.size)
    mascara = Image.open(plantilla).convert("L").resize(fondo.size)
    mascara = mascara.filter(ImageFilter.GaussianBlur(2))   # suaviza el borde
    return Image.composite(lena, fondo, mascara)             # blanco = Lena, negro = fondo

if __name__ == "__main__":
    fig, ax = plt.subplots(3, 4, figsize=(14, 7.5))
    for i in range(4):
        ax[0, i].imshow(Image.open("imagenes/" + originales[i]))
        ax[1, i].imshow(efecto("imagenes/" + originales[i], "imagenes/" + plantillas[i]))
        ax[2, i].imshow(Image.open("imagenes/" + plantillas[i]), cmap="gray")
    ax[0, 0].set_ylabel("Originales")
    ax[1, 0].set_ylabel("Procesadas")
    ax[2, 0].set_ylabel("Plantillas")
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.show()
