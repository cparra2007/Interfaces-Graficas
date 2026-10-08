import runpy

# cierra cada ventana para pasar a la siguiente
for s in ["p1_momentos", "p2_histograma_pil", "p3_planos_color", "p4_efectos_mascara",
          "p5_planos_neurona", "p6_histograma_lena", "p7_colorear_gris"]:
    print("\n#####", s)
    runpy.run_module(s, run_name="__main__")
