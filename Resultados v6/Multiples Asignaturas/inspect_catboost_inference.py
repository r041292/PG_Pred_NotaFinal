import json
import pathlib
import sys

sys.stdout.reconfigure(encoding="utf-8")

NB_PATH = pathlib.Path(
    r"C:\Users\Rubiel\OneDrive - Universidad del Norte\Maestria\Proyecto de grado\Prereq\Resultados v6\Multiples Asignaturas\rev_multiples_asig_rev_ia.ipynb"
)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    ns = {"__name__": "__main__"}

    for idx in [3, 5, 7, 9, 11, 13, 15, 18]:
        src = "".join(nb["cells"][idx]["source"])
        exec(compile(src, f"<cell {idx}>", "exec"), ns)

    cols_to_excl = [
        "Nombre_Programa",
        "_ Matricula detalle para analisis.Prof_Codigo",
        "_ Matricula detalle para analisis.Sexo",
        "_ Matricula detalle para analisis.Procedencia Categoria",
    ]

    df_usar = ns["df_historial"][
        (
            (ns["df_historial"]["Observacion_Prerrequisito"] == "Prerrequisito cumplido")
            | (ns["df_historial"]["Observacion_Prerrequisito"] == "No tiene pre requisito")
        )
        & (ns["df_historial"]["Cod materia curso"].isin(["ECO2120"]))
    ].copy()

    df_usar = ns["arreglar_comas_por_puntos"](df_usar, cols_to_excl)
    df_usar = ns["limpiar_dataframe"](df_usar, True)
    df_usar = df_usar[df_usar["Periodo"] == 202530]
    if "_ Matricula detalle para analisis.DPTO Asignatura" in df_usar.columns:
        df_usar["DPTO Asignatura"] = df_usar["_ Matricula detalle para analisis.DPTO Asignatura"]

    df_asig = df_usar[df_usar["Cod materia curso"] == "ECO2120"].copy()
    df_asig, col_usar, var_obj = ns["renombrar_columnas"](df_asig, tiene_prereq=True)
    lista_pr, df_asig = ns["columnas_prereq_validas_ext"](
        df_asig, ns["df_historial_asignaturas_nombres"], 0.8
    )
    col_usar = col_usar + lista_pr
    var_obj_clas = "Retiro_Asignatura_Cat"
    df_asig, col_usar, _ = ns["preparar_dataframe_para_modelado"](
        df_asig, col_usar, var_obj, var_obj_clas
    )
    df_asig = ns["cambiar_a_category"](
        df_asig,
        ["programa", "sexo", "procedencia_categoria", "profesor_codigo", "Tipo_colegio", "Tipo_calendario"],
    )

    modelo_reg = ns["cargar_modelo_catboost_regresion"]("", "ECO2120", variant="main")
    modelo_cls = ns["cargar_modelo_catboost_clasificacion"]("", "ECO2120", variant="main")
    features_reg = ns["cargar_features_modelo"](
        "ECO2120", "catboost", "main", "prediccion_nota", modelo_cargado=modelo_reg
    )
    features_cls = ns["cargar_features_modelo"](
        "ECO2120", "catboost", "main", "prediccion_retiro", modelo_cargado=modelo_cls
    )

    if not features_reg:
        try:
            features_reg = list(modelo_reg.get_feature_names())
        except Exception:
            features_reg = [c for c in df_asig.columns if c not in [var_obj, var_obj_clas]]
    if not features_cls:
        try:
            features_cls = list(modelo_cls.get_feature_names())
        except Exception:
            features_cls = list(features_reg)

    print("cat idx reg:", modelo_reg.get_cat_feature_indices())
    print("cat idx cls:", modelo_cls.get_cat_feature_indices())
    print("num feat reg:", len(features_reg), "num feat cls:", len(features_cls))

    xreg_df = ns["alinear_dataframe_a_modelo"](df_asig, features_reg, fill_value=float("nan"))
    xcls_df = ns["alinear_dataframe_a_modelo"](df_asig, features_cls, fill_value=float("nan"))

    for nombre, modelo, xdf, feats in [
        ("reg", modelo_reg, xreg_df, features_reg),
        ("cls", modelo_cls, xcls_df, features_cls),
    ]:
        print(f"--- {nombre} ---")
        cat_cols = [c for c in xdf.columns if str(xdf[c].dtype) == "category" or xdf[c].dtype == "object"]
        print("categorical cols in Xdf:", cat_cols)
        try:
            pred = modelo.predict(xdf)
            print(nombre, "direct_predict_ok", len(pred))
        except Exception as e:
            print(nombre, "direct_predict_error", repr(e))
        try:
            xnum, _ = ns["preparar_X_numerico"](xdf, feats)
            pred2 = modelo.predict(xnum)
            print(nombre, "numeric_predict_ok", len(pred2))
        except Exception as e:
            print(nombre, "numeric_predict_error", repr(e))


if __name__ == "__main__":
    main()
