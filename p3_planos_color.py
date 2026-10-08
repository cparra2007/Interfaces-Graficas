import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def planos(ruta):
    im = Image.open(ruta).convert("RGB")
    r, g, b = im.split()
    gris = im.convert("L")
    cero = np.zeros_like(np.array(r))

    # planos a color
    fig, ax = plt.subplots(2, 3, figsize=(13, 6))
    ax[0, 0].imshow(im);    ax[0, 0].set_title("Original")
    ax[0, 1].imshow(gris, cmap="gray"); ax[0, 1].set_title("Gris")
    ax[0, 2].axis("off")
    ax[1, 0].imshow(np.dstack([np.array(r), cero, cero])); ax[1, 0].set_title("Plano R")
    ax[1, 1].imshow(np.dstack([cero, np.array(g), cero])); ax[1, 1].set_title("Plano G")
    ax[1, 2].imshow(np.dstack([cero, cero, np.array(b)])); ax[1, 2].set_title("Plano B")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.show()

    # planos como intensidad
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
    for a, p, t in zip(ax, (r, g, b), "RGB"):
        a.imshow(p, cmap="gray")
        a.set_title("Plano " + t)
        a.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    planos("imagenes/flores.png")
