"""
estilo_tablas.py
=====================================
Convenciones comunes de las tablas del documento. Cada `tabla_*.py` sigue
escribiendo su propio LaTeX; este modulo solo centraliza lo que debe ser
identico en todas:

    FUENTE -- texto unico de la fuente al final de cada nota.
    aplicar_signo_menos -- convierte el guion de los numeros negativos
                           ("-0.007") en signo menos matematico ("$-$0.007"),
                           igual que en el texto.

Estandar de titulos y notas (aplicado a mano en cada script):
    - Titulo: "Que se muestra, definicion de pobreza, periodo." con punto
      final, sin definiciones ni abreviaturas (esas van en la nota).
    - Definicion de pobreza: "pobreza monetaria" / "pobreza
      multidimensional (IPM)"; periodos con flecha (2013$\\to$2016).
    - Espanol: "frente a" (no "vs."), "conjunto de prueba" (no
      holdout/test), "base de datos" (no dataset).
    - Nota: bloque minipage debajo, "\\footnotesize \\textit{Nota:}",
      empieza con mayuscula y termina con FUENTE.
    - Formato: table[H], \\footnotesize, booktabs; \\resizebox solo si la
      tabla no cabe de otra forma; el entorno completo dentro del.tex.
"""

import re

FUENTE = "Fuente: cálculos propios."

# Guion seguido de digito, que no sea parte de una palabra (ResNet-50), de
# un rango con raya (1--2), de un rango de columnas (\cmidrule{2-5}) ni de
# algo ya en modo matematico ($-0.006$).
_GUION_NUMERO = re.compile(r"(?<![\w\-$])-(?=\d)")


def aplicar_signo_menos(tex: str) -> str:
    return _GUION_NUMERO.sub("$-$", tex)
