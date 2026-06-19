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

    code_cells = [3, 5, 7, 9, 11, 13, 15, 18, 19, 20]
    for idx in code_cells:
        src = "".join(nb["cells"][idx]["source"])
        if not src.strip():
            continue
        print(f">>> Ejecutando cell {idx + 1}")
        exec(compile(src, f"<cell {idx + 1}>", "exec"), ns)

    print(">>> Preparando df_usar solo para IGL4994")
    periodo_hasta_donde_analizar = 202510
    cols_to_excl = [
        "Nombre_Programa",
        "_ Matricula detalle para analisis.Prof_Codigo",
        "_ Matricula detalle para analisis.Sexo",
        "_ Matricula detalle para analisis.Procedencia Categoria",
    ]
    asig_a_usar = ["IGL4994"]

    df_usar = ns["df_historial"][
        (
            (ns["df_historial"]["Observacion_Prerrequisito"] == "Prerrequisito cumplido")
            | (ns["df_historial"]["Observacion_Prerrequisito"] == "No tiene pre requisito")
        )
        & (ns["df_historial"]["Cod materia curso"].isin(asig_a_usar))
    ].copy()

    df_usar = ns["arreglar_comas_por_puntos"](df_usar, cols_to_excl)
    df_usar = ns["limpiar_dataframe"](df_usar, True)
    df_usar = df_usar[df_usar["Periodo"] <= periodo_hasta_donde_analizar]
    df_usar, asig_a_usar = ns["filtrar_asignaturas_por_matricula_minima"](
        df_usar, asig_a_usar, matricula_minima=1
    )
    ns["df_usar"] = df_usar

    print("Filas df_usar:", len(df_usar))
    print("Asignaturas finales:", asig_a_usar)

    if "IGL4994" not in asig_a_usar:
        raise RuntimeError(
            "IGL4994 no quedó disponible en asig_a_usar después de la preparación."
        )

    print(">>> Ejecutando correr_modelos_multi_por_asignatura solo para IGL4994")
    (
        df_resultados_final_multi,
        df_resultados_stats_multi,
        df_resultados_class_retiro_stats_multi,
    ) = ns["correr_modelos_multi_por_asignatura"](df_usar, "main")

    print(">>> PRUEBA_OK")
    print("Filas resultado final:", len(df_resultados_final_multi))
    print("Columnas resultado final:", len(df_resultados_final_multi.columns))
    print(
        "Contiene columnas de prediccion:",
        [
            c
            for c in [
                "Prediccion_XGB",
                "Prediccion_RF",
                "Prediccion_CAT",
                "Clasificacion_XGB",
                "Clasificacion_RF",
                "Clasificacion_CAT",
            ]
            if c in df_resultados_final_multi.columns
        ],
    )
    print(
        "Contiene columnas interpretacion:",
        [
            c
            for c in [
                "interpretacion_general_xgb",
                "interpretacion_registro_xgb",
                "interpretacion_general_cat",
                "interpretacion_registro_cat",
            ]
            if c in df_resultados_final_multi.columns
        ],
    )
    print("Primeras 30 columnas:")
    print(df_resultados_final_multi.columns.tolist()[:30])


if __name__ == "__main__":
    main()
