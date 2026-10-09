import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def histogramas(ruta):
    im = Image.open(ruta).convert("RGB")
    a = np.array(im)

    # histograma RGB y valor mas repetido por plano
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(im)
    ax[0].axis("off")
    for k, (n, c) in enumerate(zip("RGB", ("red", "green", "blue"))):
        h = np.bincount(a[..., k].ravel(), minlength=256)
        ax[1].plot(h, color=c, label=n)
        print(f"Plano {n}: valor mas repetido = {h.argmax()} ({h.max()} pixeles)")
    ax[1].set_xlabel("Valores Pixeles")
    ax[1].set_ylabel("Histograma RGB")
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # histograma en gris
    g = np.array(im.convert("L"))
    h = np.bincount(g.ravel(), minlength=256)
    print(f"Gris: moda = {h.argmax()}, media = {g.mean():.1f}, desviacion = {g.std():.1f}")
    print(f"Oscuros (<85): {(g < 85).mean() * 100:.1f}% | Claros (>=170): {(g >= 170).mean() * 100:.1f}%")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].imshow(g, cmap="gray")
    ax[0].axis("off")
    ax[1].fill_between(range(256), h, color="gray")
    ax[1].set_xlabel("Valores Pixeles")
    ax[1].set_ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    histogramas("imagenes/fig_00.jpg")
