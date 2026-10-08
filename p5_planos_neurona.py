import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

UMBRAL = 30  

def planos(ruta, umbral=UMBRAL):
    im = Image.open(ruta).convert("RGB")
    a = np.array(im)
    cero = np.zeros(a.shape[:2], dtype=np.uint8)
    rojo = np.dstack([a[..., 0], cero, cero])
    verde = np.dstack([cero, a[..., 1], cero])
    azul = np.dstack([cero, cero, a[..., 2]])

    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    for x, img, t in zip(ax, (im, rojo, verde, azul), ("Original", "Plano Red", "Plano Green", "Plano Blue")):
        x.imshow(img)
        x.set_title(t)
        x.axis("off")
    plt.tight_layout()

    total = a.shape[0] * a.shape[1]
    for k, n in enumerate("RGB"):
        area = int((a[..., k] > umbral).sum())
        print(f"Plano {n}: area = {area} px ({100 * area / total:.2f} %)")
    plt.show()

if __name__ == "__main__":
    planos("imagenes/fig_05.jpg")
