from PIL import Image
import matplotlib.pyplot as plt

def histograma(ruta):
    im = Image.open(ruta).convert("RGB")
    h = im.histogram()                      
    r, g, b = h[0:256], h[256:512], h[512:768]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(im)
    ax[0].axis("off")
    ax[1].plot(r, "r", label="R")
    ax[1].plot(g, "g", label="G")
    ax[1].plot(b, "b", label="B")
    ax[1].set_title("Histograma")
    ax[1].set_xlabel("Valor del pixel")
    ax[1].set_ylabel("Frecuencia")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    histograma("imagenes/mono.png")
