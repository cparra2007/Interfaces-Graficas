import math
import cv2
import numpy as np
import pygame
from PIL import Image


def detectar_bloques(imagen_pil, solo_oscuros):
    """Devuelve los 4 bloques mas grandes de la imagen, de izquierda a derecha."""
    gris = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2GRAY)
    mascara = (gris < 60) if solo_oscuros else (gris < 245)
    mascara = mascara.astype(np.uint8) * 255
    # apertura: borra marcos, ejes y texto; deja solo los bloques grandes
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:4]
    return sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])


# 1) Importar figuras y plantilla (PIL)
figuras = Image.open("figuras.png").convert("RGB")
plantilla = Image.open("plantilla.png").convert("RGB")

# 2) Recortar f1..f4 de la imagen de figuras
fotos = []
for c in detectar_bloques(figuras, solo_oscuros=False):
    x, y, w, h = cv2.boundingRect(c)
    fotos.append(figuras.crop((x, y, x + w, y + h)))

# 3) Ubicar los 4 cuadrados negros de la plantilla
cuadrados = detectar_bloques(plantilla, solo_oscuros=True)

# 4) Redimensionar, rotar y pegar cada foto en su cuadrado
for foto, c in zip(fotos, cuadrados):
    caja = cv2.boxPoints(cv2.minAreaRect(c))
    suma = caja.sum(axis=1)
    dif = caja[:, 1] - caja[:, 0]
    arriba_izq = caja[np.argmin(suma)]
    abajo_der = caja[np.argmax(suma)]
    arriba_der = caja[np.argmin(dif)]
    abajo_izq = caja[np.argmax(dif)]

    ancho = math.dist(arriba_izq, arriba_der)
    alto = math.dist(arriba_izq, abajo_izq)
    angulo = math.degrees(math.atan2(arriba_der[1] - arriba_izq[1],
                                     arriba_der[0] - arriba_izq[0]))
    centro_x, centro_y = (arriba_izq + arriba_der + abajo_der + abajo_izq) / 4

    foto = foto.convert("RGBA").resize((round(ancho), round(alto)))   # redimensionar
    foto = foto.rotate(-angulo, expand=True)                          # rotar
    plantilla.paste(foto, (round(centro_x - foto.width / 2),
                           round(centro_y - foto.height / 2)), foto)  # pegar

# 5) Mostrar el resultado con Pygame
pygame.init()
pantalla = pygame.display.set_mode(plantilla.size)
pygame.display.set_caption("Proyecto #1 - Parte 2")
superficie = pygame.image.fromstring(plantilla.tobytes(), plantilla.size, "RGB")

corriendo = True
while corriendo:
    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False
    pantalla.blit(superficie, (0, 0))
    pygame.display.flip()
pygame.quit()