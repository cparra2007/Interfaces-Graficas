
import math
import sys
import cv2
import numpy as np
import pygame
from PIL import Image

FIGURAS = "figuras.png"
PLANTILLA = "plantilla.png"
SALIDA = "salida.png"



def abrir_rgb(ruta):
    try:
        im = Image.open(ruta).convert("RGBA")
    except FileNotFoundError:
        sys.exit(f"No encuentro '{ruta}' en la carpeta del script.")
    fondo = Image.new("RGBA", im.size, (255, 255, 255, 255))
    fondo.alpha_composite(im)
    return fondo.convert("RGB")


def detectar_bloques(pil_img, solo_oscuros, n=4, k=25):

    gris = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    mask = (gris < 60) if solo_oscuros else (gris < 245)
    mask = mask.astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:n]
    if len(cnts) < n:
        sys.exit(f"Solo encontre {len(cnts)} bloques (se esperaban {n}). Ajusta k o los umbrales.")
    return sorted(cnts, key=lambda c: cv2.boundingRect(c)[0])


def esquinas(cnt):
    """Esquinas del rectangulo rotado: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    box = cv2.boxPoints(cv2.minAreaRect(cnt))
    s = box.sum(axis=1)
    d = box[:, 1] - box[:, 0]            # y - x
    tl, br = box[np.argmin(s)], box[np.argmax(s)]
    tr, bl = box[np.argmin(d)], box[np.argmax(d)]
    return tl, tr, br, bl


def main():
    figuras = abrir_rgb(FIGURAS)
    plantilla = abrir_rgb(PLANTILLA)

    fotos = []
    for c in detectar_bloques(figuras, solo_oscuros=False):
        x, y, w, h = cv2.boundingRect(c)
        fotos.append(figuras.crop((x, y, x + w, y + h)))

    destinos = detectar_bloques(plantilla, solo_oscuros=True)

    salida = plantilla.copy()
    for i, (foto, cnt) in enumerate(zip(fotos, destinos)):
        tl, tr, br, bl = esquinas(cnt)
        ancho = math.dist(tl, tr)
        alto = math.dist(tl, bl)
        ang = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))  # horario (y hacia abajo)
        cx, cy = (tl + tr + br + bl) / 4

        img = foto.convert("RGBA").resize((round(ancho) + 2, round(alto) + 2), Image.LANCZOS)
        img = img.rotate(-ang, expand=True, resample=Image.BICUBIC)
        pos = (round(cx - img.width / 2), round(cy - img.height / 2))
        salida.paste(img, pos, img)      # la mascara alfa evita pegar las esquinas vacias
        print(f"f{i+1}: lado={ancho:.0f}x{alto:.0f}px  angulo={ang:.1f} grados  centro=({cx:.0f},{cy:.0f})")

    salida.save(SALIDA)
    print(f"Guardado: {SALIDA}")

    # 4) mostrar con Pygame
    pygame.init()
    ancho_v, alto_v = salida.size
    esc = min(1.0, 1100 / ancho_v, 800 / alto_v)
    vista = salida.resize((int(ancho_v * esc), int(alto_v * esc)), Image.LANCZOS)
    pantalla = pygame.display.set_mode(vista.size)
    pygame.display.set_caption("Proyecto #1 - Parte 2 (ESC para salir)")
    surf = pygame.image.fromstring(vista.tobytes(), vista.size, "RGB")
    reloj = pygame.time.Clock()
    corriendo = True
    while corriendo:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                corriendo = False
        pantalla.blit(surf, (0, 0))
        pygame.display.flip()
        reloj.tick(30)
    pygame.quit()


if __name__ == "__main__":
    main()