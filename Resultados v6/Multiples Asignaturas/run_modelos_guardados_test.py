import json
import os
import pathlib
import sys

os.environ["MPLBACKEND"] = "Agg"
sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = pathlib.Path(
    r"C:\Users\Rubiel\OneDrive - Universidad del Norte\Maestria\Proyecto de grado\Prereq\Resultados v6\Multiples Asignaturas\rev_multiples_asig_rev_ia.ipynb"
)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}

    code_cells = [3, 5, 7, 9, 11, 13, 15, 18]
    for idx in code_cells:
        src = "".join(nb["cells"][idx]["source"])
        if not src.strip():
            continue
        print(f">>> Ejecutando cell {idx + 1}")
        exec(compile(src, f"<cell {idx + 1}>", "exec"), ns)

    ruta_modelos_guardados = ns["get_model_dir"]("xgboost", "main", "prediccion_nota")
    asignaturas_mod_guardados = ns["check_asignaturas_en_carpeta"](ruta_modelos_guardados)
    print("Asignaturas detectadas:", asignaturas_mod_guardados)

    if "ECO2120" not in asignaturas_mod_guardados:
        raise RuntimeError("ECO2120 no fue detectada en la carpeta de modelos guardados.")

    print(">>> Preparando df_usar solo para ECO2120")
    periodo_a_evaluar = 202530
    cols_to_excl = [
        "Nombre_Programa",
        "_ Matricula detalle para analisis.Prof_Codigo",
        "_ Matricula detalle para analisis.Sexo",
        "_ Matricula detalle para analisis.Procedencia Categoria",
    ]
    asig_a_usar = ["ECO2120"]

    df_usar = ns["df_historial"][
        (
            (ns["df_historial"]["Observacion_Prerrequisito"] == "Prerrequisito cumplido")
            | (ns["df_historial"]["Observacion_Prerrequisito"] == "No tiene pre requisito")
        )
        & (ns["df_historial"]["Cod materia curso"].isin(asig_a_usar))
    ].copy()

    df_usar = ns["arreglar_comas_por_puntos"](df_usar, cols_to_excl)
    df_usar = ns["limpiar_dataframe"](df_usar, True)

    if "Periodo" in df_usar.columns:
        df_usar = df_usar[df_usar["Periodo"] == periodo_a_evaluar]
    if "_ Matricula detalle para analisis.DPTO Asignatura" in df_usar.columns:
        df_usar["DPTO Asignatura"] = df_usar["_ Matricula detalle para analisis.DPTO Asignatura"]

    if df_usar.empty:
        raise RuntimeError("df_usar quedó vacío para ECO2120 después de la preparación.")

    ns["df_usar"] = df_usar
    print("Filas df_usar:", len(df_usar))
    print("Columnas df_usar:", len(df_usar.columns))

    carpeta_salida = pathlib.Path(ns["NOTEBOOK_DIR"]) / f"Resultados_Ejecucion_Modelo_{ns['MODEL_VERSION']}"
    existentes_antes = set()
    if carpeta_salida.exists():
        existentes_antes = {p.name for p in carpeta_salida.glob("*.csv")}

    print(">>> Ejecutando usar_modelos_guardados_xg_cat_por_asignatura solo para ECO2120")
    df_resultados_final, df_errores, _ = ns["usar_modelos_guardados_xg_cat_por_asignatura"](df_usar)

    print(">>> PRUEBA_OK")
    print("Filas resultado final:", len(df_resultados_final))
    print("Columnas resultado final:", len(df_resultados_final.columns))
    print("Errores:", 0 if df_errores.empty else len(df_errores))
    print(
        "Columnas clave presentes:",
        [
            c
            for c in [
                "Clasificacion_XGB",
                "Prediccion_XGB",
                "Prediccion_final_XGB",
                "Clasificacion_CAT",
                "Prediccion_CAT",
                "Prediccion_final_CAT",
                "interpretacion_general_xgb",
                "interpretacion_registro_xgb",
                "interpretacion_general_cat",
                "interpretacion_registro_cat",
            ]
            if c in df_resultados_final.columns
        ],
    )

    existentes_despues = set()
    if carpeta_salida.exists():
        existentes_despues = {p.name for p in carpeta_salida.glob("*.csv")}
    nuevos = sorted(existentes_despues - existentes_antes)
    print("Archivos nuevos en carpeta de salida:", nuevos)

    if not nuevos:
        raise RuntimeError("No se detectó archivo nuevo en la carpeta de Resultados_Ejecucion_Modelo.")


if __name__ == "__main__":
    main()
