import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# x = columna, y = fila; figura = pixeles no blancos
def cargar(ruta):
    rgb = np.array(Image.open(ruta).convert("RGB"))
    f = (rgb.min(axis=2) < 200).astype(float)
    return rgb, f

# momento bruto m_pq
def m(f, p, q):
    y, x = np.indices(f.shape)
    return np.sum(x**p * y**q * f)

# momento central mu_pq
def mu(f, p, q):
    xc = m(f, 1, 0) / m(f, 0, 0)
    yc = m(f, 0, 1) / m(f, 0, 0)
    y, x = np.indices(f.shape)
    return np.sum((x - xc)**p * (y - yc)**q * f)

# momento central normalizado eta_pq
def eta(f, p, q):
    return mu(f, p, q) / mu(f, 0, 0)**(1 + (p + q) / 2)

def mostrar(img, titulo, lineas):
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.title(titulo, fontweight="bold")
    fig.text(0.5, 0.03, "\n".join(lineas), ha="center", va="bottom", family="monospace")
    plt.subplots_adjust(bottom=0.08 + 0.055 * len(lineas))
    plt.show()

def figura_1a(ruta):
    rgb, f = cargar(ruta)
    area = int(f.sum())
    ys, xs = np.nonzero(f)
    cx, cy = xs.mean(), ys.mean()
    cxm, cym = m(f, 1, 0) / m(f, 0, 0), m(f, 0, 1) / m(f, 0, 0)

    # cruz sobre la imagen (ampliada x6)
    e = 6
    img = cv2.resize(rgb, None, fx=e, fy=e, interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(img, (int(cx * e + e / 2), int(cy * e + e / 2)), (0, 0, 0), cv2.MARKER_CROSS, 40, 3)

    print("--- Figura 1.a ---")
    print(f"a) Area = {area}")
    print(f"b) Centroide = ({cx:.3f}, {cy:.3f})")
    print(f"c) Centroide con momentos = ({cxm:.3f}, {cym:.3f})")
    mostrar(img, "Figura 1.a", [f"a) Area = {area} px",
                                f"b) Centroide = ({cx:.2f}, {cy:.2f})",
                                f"c) Centroide (momentos) = ({cxm:.2f}, {cym:.2f})"])

def figura_1b(ruta, p=2, q=3):
    rgb, f = cargar(ruta)
    a, b, c = m(f, p, q), mu(f, p, q), eta(f, p, q)
    print(f"--- Figura 1.b (p={p}, q={q}) ---")
    print(f"a) m{p}{q} = {a:.6e}")
    print(f"b) mu{p}{q} = {b:.6e}")
    print(f"c) eta{p}{q} = {c:.6e}")
    mostrar(rgb, "Figura 1.b", [f"a) Momento m({p},{q}) = {a:.4e}",
                                f"b) Momento central mu({p},{q}) = {b:.4e}",
                                f"c) Mom. central normalizado eta({p},{q}) = {c:.4e}"])

def figura_1c(ruta):
    rgb, f = cargar(ruta)
    n20, n02, n11 = eta(f, 2, 0), eta(f, 0, 2), eta(f, 1, 1)
    n30, n12, n21, n03 = eta(f, 3, 0), eta(f, 1, 2), eta(f, 2, 1), eta(f, 0, 3)
    H1 = n20 + n02
    H2 = (n20 - n02)**2 + 4 * n11**2
    H3 = (n30 - 3 * n12)**2 + (3 * n21 - n03)**2
    hu = cv2.HuMoments(cv2.moments(f)).flatten()   # comprobacion con OpenCV
    print("--- Figura 1.c (Hu) ---")
    print(f"H1 = {H1:.6e} (OpenCV {hu[0]:.6e})")
    print(f"H2 = {H2:.6e} (OpenCV {hu[1]:.6e})")
    print(f"H3 = {H3:.6e} (OpenCV {hu[2]:.6e})")
    mostrar(rgb, "Figura 1.c - Momentos de Hu", [f"H1 = {H1:.6e}", f"H2 = {H2:.6e}", f"H3 = {H3:.6e}"])

if __name__ == "__main__":
    figura_1a("imagenes/a.png")
    figura_1b("imagenes/b.png")
    figura_1c("imagenes/c.png")
