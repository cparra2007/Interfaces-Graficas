
import sys
import csv
import cv2
import numpy as np

RUTA = sys.argv[1] if len(sys.argv) > 1 else "vocales.png"
NOMBRES = ["A", "E", "I", "O", "U"]


def cargar_gris(ruta):
    """Lee la imagen (aplanando transparencia sobre blanco) y la pasa a gris."""
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"No pude abrir '{ruta}'. Ponla en la misma carpeta del script.")
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        img = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def segmentar(gris):
    """Binariza (letras = 255) y devuelve una mascara por cada vocal, de izq. a der."""
    _, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    if bw.mean() > 127:                      # si el fondo quedo como objeto, invertir
        bw = 255 - bw
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_min = 0.002 * bw.size               # descarta ruido
    cnts = [c for c in cnts if cv2.contourArea(c) > area_min]
    cnts.sort(key=lambda c: cv2.boundingRect(c)[0])
    if len(cnts) != 5:
        print(f"[aviso] Se detectaron {len(cnts)} objetos (se esperaban 5).")
    masks = []
    for c in cnts:
        m = np.zeros_like(bw)
        cv2.drawContours(m, [c], -1, 255, thickness=cv2.FILLED)
        masks.append(cv2.bitwise_and(m, bw))  # conserva huecos internos (A, O...)
    return masks


def hu(mask):
    m = cv2.moments(mask, binaryImage=True)
    return cv2.HuMoments(m).flatten()


def log_hu(h):
    """-sign(h) * log10(|h|)  (estandar para comparar magnitudes tan distintas)."""
    h = np.where(h == 0, 1e-30, h)
    return -np.sign(h) * np.log10(np.abs(h))


def transformar(mask, ang=45, esc=0.6):
    """Rota y escala una mascara (para comprobar la invarianza)."""
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, esc)
    r = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_LINEAR)
    return (r > 127).astype(np.uint8) * 255   # re-binarizar tras interpolar


def imprimir_tabla(titulo, filas, nombres):
    print(f"\n{titulo}")
    cab = "      " + "".join(f"Log(H{i})".rjust(11) for i in range(1, 8))
    print(cab)
    print("-" * len(cab))
    for n, f in zip(nombres, filas):
        print(f"  {n}   " + "".join(f"{v:11.4f}" for v in f))


def main():
    gris = cargar_gris(RUTA)
    masks = segmentar(gris)
    nombres = NOMBRES[:len(masks)]

    H = [hu(m) for m in masks]
    L = [log_hu(h) for h in H]

    imprimir_tabla("Momentos de Hu (valores crudos)", H, nombres)
    imprimir_tabla("TABLA RESUMEN  Log(Hu's)", L, nombres)

    # --- Prueba de invarianza: misma vocal rotada 45 y a escala 0.6 ---
    L2 = [log_hu(hu(transformar(m))) for m in masks]
    imprimir_tabla("Log(Hu) tras rotar 45 grados y escalar x0.6", L2, nombres)
    dif = [np.abs(a - b) for a, b in zip(L, L2)]
    imprimir_tabla("|diferencia| (cercano a 0 = invariante)", dif, nombres)

    with open("tabla_resumen_hu.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Vocal"] + [f"Log(H{i})" for i in range(1, 8)])
        for n, fila in zip(nombres, L):
            w.writerow([n] + [f"{v:.6f}" for v in fila])
    print("\nTabla guardada en tabla_resumen_hu.csv")


if __name__ == "__main__":
    main()