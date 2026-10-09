import cv2 #nos sirver para leer la imagen y procesarla
import numpy as np

# 1) Cargar imagen y pasar a gris
img = cv2.imread("vocales.png")
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2) Binarizar (letras en blanco, fondo en negro)
_, bw = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 3) Encontrar cada vocal y ordenarlas de izquierda a derecha
contornos, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos = [c for c in contornos if cv2.contourArea(c) > 100]   # descarta ruido
contornos.sort(key=lambda c: cv2.boundingRect(c)[0])

vocales = ["A", "E", "I", "O", "U"]

# 4) Hu(1..7) de cada vocal y Log(Hu)
print("Tabla Resumen Log(Hu's)")
print("   " + "".join(f"Log(H{i})".rjust(11) for i in range(1, 8)))

for nombre, c in zip(vocales, contornos):
    mascara = np.zeros_like(bw)
    cv2.drawContours(mascara, [c], -1, 255, cv2.FILLED)
    mascara = cv2.bitwise_and(mascara, bw)

    momentos = cv2.moments(mascara, binaryImage=True) # calcular momentos
    hu = cv2.HuMoments(momentos).flatten()
    hu = np.where(hu == 0, 1e-30, hu)          # evita log(0) en letras simetricas
    log_hu = -np.sign(hu) * np.log10(np.abs(hu))

    print(f" {nombre} " + "".join(f"{v:11.4f}" for v in log_hu))