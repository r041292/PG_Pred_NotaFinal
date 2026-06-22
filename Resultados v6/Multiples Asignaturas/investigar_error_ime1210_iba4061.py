"""Diagnostico temporal del MegaPipeline; no modifica el notebook."""
import json
import os
import pathlib
import sys
import traceback

os.environ["MPLBACKEND"] = "Agg"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parent
NB_PATH = ROOT / "rev_multiples_asig_rev_ia.ipynb"
ASIGNATURAS = ["IME1210", "IBA4061"]


def ejecutar_celda(nb, indice, ns):
    source = "".join(nb["cells"][indice]["source"])
    print(f">>> Ejecutando celda {indice + 1}")
    exec(compile(source, f"<celda {indice + 1}>", "exec"), ns)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}

    # Solo helpers y carga inicial: se omiten las celdas de ejecucion que limpian
    # carpetas de modelos o lanzan rutinas automaticamente.
    for indice in [3, 5, 7, 9, 11, 13]:
        ejecutar_celda(nb, indice, ns)
    for indice in [18, 19, 20]:
        ejecutar_celda(nb, indice, ns)

    df_historial = ns["df_historial"]
    cols_to_excl = [
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
    df_usar = ns["arreglar_comas_por_puntos"](df_usar, cols_to_excl)
    df_usar = ns["limpiar_dataframe"](
        df_usar, True, eliminar_filas_durante_limpieza=True
    )
    df_usar = df_usar[df_usar["Periodo"] <= 202510]
    df_usar, asignaturas_finales = ns["filtrar_asignaturas_por_matricula_minima"](
        df_usar, ASIGNATURAS, matricula_minima=200
    )
    print("Asignaturas finales:", asignaturas_finales)
    print("Filas por asignatura:\n", df_usar["Cod materia curso"].value_counts())

    for asignatura in asignaturas_finales:
        df_asig = df_usar[df_usar["Cod materia curso"] == asignatura].copy()
        tiene_prereq = (
            df_asig["Observacion_Prerrequisito"].dropna().iloc[0]
            == "Prerrequisito cumplido"
        )
        df_preparado, columnas_base, objetivo = ns["renombrar_columnas"](
            df_asig, tiene_prereq=tiene_prereq
        )
        if tiene_prereq:
            columnas_prereq, df_preparado = ns["columnas_prereq_validas_ext"](
                df_preparado, ns["df_historial_asignaturas_nombres"], 0.8
            )
        else:
            columnas_prereq = ns["columnas_prereq_validas"](df_preparado, 0.8)

        predictores = list(dict.fromkeys(columnas_base + columnas_prereq))
        df_modelo, predictores, faltantes = ns["preparar_dataframe_para_modelado"](
            df_preparado, predictores, objetivo, "Retiro_Asignatura_Cat"
        )
        print(f"\n[Diagnostico] {asignatura}")
        print("Objetivo de regresion:", objetivo)
        print("Objetivo de clasificacion: Retiro_Asignatura_Cat")
        print("Predictores finales ({}):".format(len(predictores)))
        for numero, columna in enumerate(predictores, start=1):
            print(f"  {numero}. {columna}")
        print("Valores no nulos en columnas Prereq_calculo:")
        for columna in predictores:
            if columna.startswith("Prereq_calculo"):
                print(f"  {columna}: {df_modelo[columna].notna().sum()} de {len(df_modelo)}")
        print("Columnas duplicadas en dataframe de modelo:", df_modelo.columns[df_modelo.columns.duplicated()].tolist())


if __name__ == "__main__":
    main()
