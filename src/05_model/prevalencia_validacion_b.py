"""
prevalencia_validacion_b.py
===========================
Proporción de hogares que entran en pobreza en las dos ventanas unidas
(2010->2013 y 2013->2016), Modelo B, pobreza monetaria. Es la referencia al
azar del AUC-PR en la validación agrupada por hogar y por comunidad
(`robustez_hogares_no_repetidos.py`), que une las dos ventanas.

Se guarda como resultado agregado para que `src/tabla_robustez_validacion.py`
no tenga que abrir las bases por hogar.

INPUT

    data/processed/benchmark_train_test/modelo_B_{2010_2013,2013_2016}.parquet

OUTPUT

    data/processed/benchmark_resultados/robustez_asesor_2026_10/prevalencia_validacion_b.csv
    (una fila: n_hogares, n_caen, prevalencia)

COMO CORRER

    cd src/05_model && python prevalencia_validacion_b.py
"""
import pandas as pd

import modelo_utils as mu

DATOS = mu.PROJECT_ROOT / "data" / "processed" / "benchmark_train_test"
SALIDA = mu.RESULTADOS_DIR / "robustez_asesor_2026_10" / "prevalencia_validacion_b.csv"


def main() -> None:
    y = pd.concat(
        [pd.read_parquet(DATOS / f"modelo_B_{v}.parquet", columns=["Y"]) for v in ("2010_2013", "2013_2016")]
    )["Y"]
    salida = pd.DataFrame([{"n_hogares": len(y), "n_caen": int(y.sum()), "prevalencia": y.mean()}])
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SALIDA, index=False)
    print(salida.to_string(index=False))
    print(f"Guardado: {SALIDA}")


if __name__ == "__main__":
    main()
