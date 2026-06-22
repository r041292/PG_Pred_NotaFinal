"""Diagnostico temporal de Rutina 2 para IBA4061; no modifica el notebook."""
import json
import os
from pathlib import Path
import sys
import traceback

os.environ["MPLBACKEND"] = "Agg"
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
NB_PATH = ROOT / "rev_multiples_asig_rev_ia.ipynb"
ASIGNATURA = "IME1210"


def ejecutar(nb, indice, ns):
    source = "".join(nb["cells"][indice]["source"])
    exec(compile(source, f"<celda {indice + 1}>", "exec"), ns)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__", "traceback": traceback}
    for indice in [3, 5, 7, 9, 11, 13, 18, 19, 20]:
        print(f">>> Ejecutando celda {indice + 1}")
        ejecutar(nb, indice, ns)

    cols_to_excl = [
        "Nombre_Programa",
        "_ Matricula detalle para analisis.Prof_Codigo",
        "_ Matricula detalle para analisis.Sexo",
        "_ Matricula detalle para analisis.Procedencia Categoria",
    ]
    df_usar = ns["df_historial"][
        ns["df_historial"]["Observacion_Prerrequisito"].isin(
            ["Prerrequisito cumplido", "No tiene pre requisito"]
        )
        & ns["df_historial"]["Cod materia curso"].eq(ASIGNATURA)
    ].copy()
    df_usar = ns["arreglar_comas_por_puntos"](df_usar, cols_to_excl)
    df_usar = ns["limpiar_dataframe"](
        df_usar, True, eliminar_filas_durante_limpieza=False
    )
    df_usar = df_usar[df_usar["Periodo"] == 202530]
    origen_dpto = "_ Matricula detalle para analisis.DPTO Asignatura"
    if origen_dpto in df_usar.columns:
        df_usar["DPTO Asignatura"] = df_usar[origen_dpto]
    print("Filas de inferencia IBA4061:", len(df_usar))

    # Preparación idéntica a Rutina 2 hasta obtener la base de inferencia.
    df_base, columnas_base, objetivo = ns["renombrar_columnas"](
        df_usar, tiene_prereq=True
    )
    columnas_prereq, df_base = ns["columnas_prereq_validas_ext"](
        df_base, ns["df_historial_asignaturas_nombres"], 0.8
    )
    predictores = list(dict.fromkeys(columnas_base + columnas_prereq))
    objetivo_clas = "Retiro_Asignatura_Cat"
    df_base, predictores, _ = ns["preparar_dataframe_para_modelado"](
        df_base, predictores, objetivo, objetivo_clas
    )
    df_base = ns["cambiar_a_category"](
        df_base,
        ["programa", "sexo", "procedencia_categoria", "profesor_codigo"],
    )

    def alinear_categorias_xgb(df, modelo):
        """Alinea categorías de inferencia con las serializadas por XGBoost.

        Las categorías nuevas pasan a NaN para que XGBoost las trate como missing,
        sin rechazar por completo el lote de predicción.
        """
        df = df.copy()
        categorias = modelo.get_booster().get_categories(
            export_to_arrow=True
        ).to_arrow()
        for nombre, valores in categorias:
            if valores is None or nombre not in df.columns:
                continue
            try:
                permitidos = valores.to_pylist()
            except UnicodeDecodeError:
                # Algunos dominios de texto heredados contienen bytes mal
                # codificados. Se conservan tal cual; los dominios legibles,
                # como profesor_codigo, siguen alineándose normalmente.
                print(f"[Alineacion XGB] {nombre}: dominio no decodificable; se conserva sin cambio")
                continue
            serie = df[nombre]
            no_vistos = serie.notna() & ~serie.isin(permitidos)
            if no_vistos.any():
                print(
                    f"[Alineacion XGB] {nombre}: {int(no_vistos.sum())} valores "
                    "no vistos convertidos a NaN"
                )
            df[nombre] = __import__("pandas").Categorical(
                serie.where(~no_vistos), categories=permitidos
            )
        return df

    def features(modelo, task):
        return ns["cargar_features_modelo"](
            ASIGNATURA, "xgboost" if modelo == "xgb" else "catboost", "main", task
        )

    resultados = {}

    def verificar(nombre, funcion):
        try:
            salida = funcion()
            resultados[nombre] = "OK"
            print(f"[Verificacion] {nombre}: OK ({len(salida)} predicciones)")
            print(f"[Muestra] {nombre}: {salida.head(3).to_dict(orient='list')}")
        except Exception as exc:
            resultados[nombre] = f"ERROR: {type(exc).__name__}: {exc}"
            print(f"[Verificacion] {nombre}: {resultados[nombre]}")
            traceback.print_exc()

    def xgb_clasificacion():
        modelo = ns["cargar_modelo_xg_clasificacion"]("", ASIGNATURA, variant="main")
        cols = features("xgb", "prediccion_retiro")
        X = ns["alinear_dataframe_a_modelo"](df_base, cols, fill_value=0.0)
        X = alinear_categorias_xgb(X, modelo)
        return ns["usar_xgboost_clasificacion"](X, modelo, objetivo_clas, cols)

    def xgb_regresion():
        modelo = ns["cargar_modelo_xg_regresion"]("", ASIGNATURA, variant="main")
        cols = features("xgb", "prediccion_nota")
        X = ns["alinear_dataframe_a_modelo"](df_base, cols, fill_value=0.0)
        X = alinear_categorias_xgb(X, modelo)
        return ns["usar_xgboost_regresion"](X, modelo, objetivo, cols)

    def cat_clasificacion():
        modelo = ns["cargar_modelo_catboost_clasificacion"]("", ASIGNATURA, variant="main")
        cols = features("catboost", "prediccion_retiro")
        X = ns["alinear_dataframe_a_modelo"](df_base, cols, fill_value=float("nan"))
        X, _ = ns["preparar_X_numerico"](X, cols)
        return __import__("pandas").DataFrame({"pred": modelo.predict(X)})

    def cat_regresion():
        modelo = ns["cargar_modelo_catboost_regresion"]("", ASIGNATURA, variant="main")
        cols = features("catboost", "prediccion_nota")
        X = ns["alinear_dataframe_a_modelo"](df_base, cols, fill_value=float("nan"))
        X, _ = ns["preparar_X_numerico"](X, cols)
        return __import__("pandas").DataFrame({"pred": modelo.predict(X)})

    verificar("XGBoost clasificacion", xgb_clasificacion)
    verificar("XGBoost regresion", xgb_regresion)
    verificar("CatBoost clasificacion", cat_clasificacion)
    verificar("CatBoost regresion", cat_regresion)
    print("[Resumen]", resultados)

    # Verificacion final sobre la Rutina 2 tal como quedó en el notebook.
    ejecutar(nb, 15, ns)
    resultado_final, errores, _ = ns["usar_modelos_guardados_xg_cat_por_asignatura"](
        df_usar, ns["df_historial_asignaturas_nombres"], "main"
    )
    placeholder = "[Modelo no disponible]"
    columnas_xgb = ["Clasificacion_XGB", "Prediccion_XGB", "Prediccion_final_XGB"]
    conteos = {
        columna: int(resultado_final[columna].eq(placeholder).sum())
        for columna in columnas_xgb
    }
    print("[Verificacion Rutina 2] placeholders XGB:", conteos)
    print("[Verificacion Rutina 2] errores generales:", len(errores))


if __name__ == "__main__":
    main()
