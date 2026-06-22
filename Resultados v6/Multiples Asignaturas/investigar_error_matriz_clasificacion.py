"""Diagnóstico temporal de fallos de clasificación que bloquean regresión."""
import json
import os
from pathlib import Path
import sys

os.environ["MPLBACKEND"] = "Agg"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "rev_multiples_asig_rev_ia.ipynb"
ASIGNATURAS = ["IDS0045", "TCH4000", "HIS2110", "IEI4010", "IEL4010"]


def ejecutar_celda(nb, indice, ns):
    source = "".join(nb["cells"][indice]["source"])
    exec(compile(source, f"<celda {indice + 1}>", "exec"), ns)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}
    for indice in [3, 5, 7, 9, 11, 13, 15, 18, 19, 20]:
        ejecutar_celda(nb, indice, ns)

    df_historial = ns["df_historial"]
    cols_excluidas = [
        "Nombre_Programa",
        "_ Matricula detalle para analisis.Prof_Codigo",
        "_ Matricula detalle para analisis.Sexo",
        "_ Matricula detalle para analisis.Procedencia Categoria",
    ]
    df_usar = df_historial[
        df_historial["Observacion_Prerrequisito"].isin(
            ["Prerrequisito cumplido", "No tiene pre requisito"]
        )
        & df_historial["Cod materia curso"].isin(ASIGNATURAS)
    ].copy()
    df_usar = ns["arreglar_comas_por_puntos"](df_usar, cols_excluidas)
    df_usar = ns["limpiar_dataframe"](
        df_usar, True, eliminar_filas_durante_limpieza=True
    )
    df_usar = df_usar[df_usar["Periodo"] <= 202510]

    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    for asignatura in [a for a in ASIGNATURAS if a in set(df_usar["Cod materia curso"])]:
        df = df_usar[df_usar["Cod materia curso"] == asignatura].copy()
        prereq = df["Observacion_Prerrequisito"].dropna().iloc[0] == "Prerrequisito cumplido"
        df, columnas, objetivo = ns["renombrar_columnas"](df, tiene_prereq=prereq)
        if prereq:
            cols_prereq, df = ns["columnas_prereq_validas_ext"](
                df, ns["df_historial_asignaturas_nombres"], 0.8
            )
        else:
            cols_prereq = ns["columnas_prereq_validas"](df, 0.8)
        columnas = list(dict.fromkeys(columnas + cols_prereq))
        objetivo_clas = "Retiro_Asignatura_Cat"
        df, columnas, _ = ns["preparar_dataframe_para_modelado"](
            df, columnas, objetivo, objetivo_clas
        )
        df, _ = ns["eliminar_filas_por_columna"](df)
        y = df[objetivo_clas]
        _, _, y_train, y_test = train_test_split(
            df[columnas], y, test_size=0.2, random_state=42
        )
        etiquetas_test = y_test.value_counts().sort_index().to_dict()
        matriz_unica = confusion_matrix(y_test, y_test)
        print(f"\n[Diagnostico] {asignatura}")
        print("Filas despues de limpieza:", len(df))
        print("Clases totales:", y.value_counts().sort_index().to_dict())
        print("Clases en test (random_state=42):", etiquetas_test)
        print("Forma de confusion_matrix sin labels:", matriz_unica.shape)
        print("Etiquetas fijas usadas por los graficos:", ["No Retiro", "Retiro"])
        if matriz_unica.shape != (2, 2):
            print("[Hallazgo] La matriz tiene menos de 2x2, pero el grafico recibe 2 etiquetas.")

        # Corrección propuesta: fijar las dos clases al calcular la matriz.
        # Así el tamaño de la matriz siempre coincide con las dos etiquetas.
        matriz_corregida = confusion_matrix(y_test, y_test, labels=[0, 1])
        try:
            display = ConfusionMatrixDisplay(
                confusion_matrix=matriz_corregida,
                display_labels=["No Retiro", "Retiro"],
            )
            display.plot(cmap="Blues")
            plt.close(display.figure_)
            print("[Validacion] Matriz corregida:", matriz_corregida.shape, "sin error de ticks")
        except Exception as error:
            print("[Validacion] Error inesperado con matriz corregida:", repr(error))


if __name__ == "__main__":
    main()
