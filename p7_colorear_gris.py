from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def colorear(ruta):
    gris = Image.open(ruta).convert("L")
    g = ImageOps.autocontrast(gris, cutoff=1) 
    color = ImageOps.colorize(g, black=(0, 40, 140), mid=(10, 90, 210), white=(255, 255, 255), midpoint=60)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].imshow(gris, cmap="gray")
    ax[0].set_title("Figura original en Gris")
    ax[1].imshow(color)
    ax[1].set_title("Figura coloreada")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    colorear("imagenes/sea.jpg")
