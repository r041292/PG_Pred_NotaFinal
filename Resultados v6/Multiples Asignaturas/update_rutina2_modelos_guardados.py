import json
from pathlib import Path


NB_PATH = Path(
    r"C:\Users\Rubiel\OneDrive - Universidad del Norte\Maestria\Proyecto de grado\Prereq\Resultados v6\Multiples Asignaturas\rev_multiples_asig_rev_ia.ipynb"
)


CHECK_FUNC = """## check asgianturas en carpeta

import os

def check_asignaturas_en_carpeta(ruta_carpeta_procesados):
    \"\"\"
    Devuelve los codigos de asignatura que tienen un archivo de modelo valido
    dentro de la carpeta indicada, ignorando artefactos auxiliares.
    \"\"\"
    if not os.path.exists(ruta_carpeta_procesados):
        print(\"[Resultados Funcion : INFO] \", f\"[Aviso] La carpeta '{ruta_carpeta_procesados}' no existe.\")
        return []

    extensiones_validas = {'.json', '.joblib', '.cbm', '.model'}
    sufijos_auxiliares = ('_features', '_meta', '_metricas', '_salida')
    asignaturas = []

    for nombre in os.listdir(ruta_carpeta_procesados):
        ruta_archivo = os.path.join(ruta_carpeta_procesados, nombre)
        if not os.path.isfile(ruta_archivo):
            continue

        stem, ext = os.path.splitext(nombre)
        if ext.lower() not in extensiones_validas:
            continue
        if any(stem.endswith(sufijo) for sufijo in sufijos_auxiliares):
            continue

        asignaturas.append(stem)

    asignaturas = sorted(set(asignaturas))
    archivos_df = pd.DataFrame(asignaturas, columns=['asignatura'])
    print(\"[Resultados Funcion : INFO] \", f\"[Info] Asignaturas detectadas en la carpeta de modelos: {ruta_carpeta_procesados}\")
    if 'display' in globals():
        display(archivos_df)
    else:
        print(archivos_df)

    return asignaturas
"""


PREPARAR_HELPERS = """
def preparar_dataframe_para_modelado(df_asignatura, col_usar, var_objetivo, var_objetivo_clas):
    \"\"\"
    Prepara el dataframe para modelado usando solo las columnas disponibles y
    garantizando la presencia de las variables objetivo cuando existan.
    \"\"\"
    df_preparado = df_asignatura.copy()

    if var_objetivo_clas in df_preparado.columns:
        df_preparado[var_objetivo_clas] = df_preparado[var_objetivo_clas].astype(int)

    cols_present = [c for c in col_usar if c in df_preparado.columns]
    if var_objetivo in df_preparado.columns and var_objetivo not in cols_present:
        cols_present.append(var_objetivo)
    if var_objetivo_clas in df_preparado.columns and var_objetivo_clas not in cols_present:
        cols_present.append(var_objetivo_clas)

    faltantes = [c for c in col_usar if c not in df_preparado.columns]
    if faltantes:
        print("[Resultados Funcion : INFO] ", f'[Aviso] Algunas columnas de col_usar no existen y se omiten: {faltantes}')

    df_preparado = df_preparado[cols_present].copy()
    print("[Resultados Funcion : INFO] ", f'[Info] df_usar_filtrado reducido a {df_preparado.shape[1]} columnas y {len(df_preparado)} filas')

    col_usar_validas = [c for c in col_usar if c in df_preparado.columns]
    return df_preparado, col_usar_validas, faltantes

def anexar_columnas_por_indice(df_destino, df_fuente, columnas):
    columnas_presentes = [c for c in columnas if c in df_fuente.columns]
    for col in columnas_presentes:
        df_destino[col] = df_fuente[col].reindex(df_destino.index)
    return df_destino

def limitar_predicciones_nota(df, columnas_prediccion):
    for col in columnas_prediccion:
        if col in df.columns:
            df[col] = df[col].round(2)
            df[col] = df[col].apply(lambda x: -1 if pd.notna(x) and x < 0 else x)
            df[col] = df[col].apply(lambda x: 5 if pd.notna(x) and x > 5 else x)
    return df
"""


USAR_MODELOS_FUNC = """def usar_modelos_guardados_xg_cat_por_asignatura(df_usar, variant_modelo='main'):
    col_usar = []
    var_objetivo = ''
    df_resultados_final = pd.DataFrame()
    df_errores = pd.DataFrame()
    df_pred_cat_ultimo = None
    asig_a_usar = df_usar['Cod materia curso'].unique().tolist()
    placeholder_modelo = '[Modelo no disponible]'

    ruta_modelos_reg_xgb = get_model_dir('xgboost', variant_modelo, 'prediccion_nota')
    ruta_modelos_cls_xgb = get_model_dir('xgboost', variant_modelo, 'prediccion_retiro')
    ruta_modelos_reg_cat = get_model_dir('catboost', variant_modelo, 'prediccion_nota')
    ruta_modelos_cls_cat = get_model_dir('catboost', variant_modelo, 'prediccion_retiro')

    def _resolver_features(asignatura, modelo_nombre, task, modelo_cargado, fallback_cols, excluir=None):
        excluir = set(excluir or [])
        features = cargar_features_modelo(
            asignatura,
            modelo_nombre,
            variant_modelo,
            task,
            modelo_cargado=modelo_cargado
        )
        if not features:
            features = _infer_features_from_model(modelo_cargado, modelo_nombre)
        if not features:
            features = [c for c in fallback_cols if c not in excluir]

        features = [c for c in features if c not in excluir]
        features = list(dict.fromkeys(features))
        return features

    def _inyectar_columnas(df_destino, df_origen, columnas):
        for columna in columnas:
            if columna in df_origen.columns:
                df_destino.loc[df_origen.index, columna] = df_origen[columna]
        return df_destino

    def _marcar_no_disponible(df_destino, columnas, indices=None):
        if indices is None:
            indices = df_destino.index
        for columna in columnas:
            df_destino.loc[indices, columna] = placeholder_modelo
        return df_destino

    def _clip_predicciones(df_pred, columna):
        if columna not in df_pred.columns:
            return df_pred
        serie = pd.to_numeric(df_pred[columna], errors='coerce').round(2)
        serie = serie.clip(lower=-1, upper=5)
        df_pred[columna] = serie
        return df_pred

    def _normalizar_columna_mixta(serie, placeholder='[Modelo no disponible]'):
        def _fmt(valor):
            if pd.isna(valor):
                return np.nan
            if isinstance(valor, str):
                return valor if valor == placeholder else valor
            try:
                return round(float(valor), 2)
            except Exception:
                return valor
        return serie.apply(_fmt)

    def _calcular_prediccion_final(df_pred, col_clas, col_pred, col_final):
        if col_clas not in df_pred.columns or col_pred not in df_pred.columns:
            df_pred[col_final] = placeholder_modelo
            return df_pred

        serie_clas = df_pred[col_clas]
        serie_pred = df_pred[col_pred]

        if pd.api.types.is_object_dtype(serie_clas) or pd.api.types.is_object_dtype(serie_pred):
            mascara_placeholder = (
                serie_clas.astype(str).eq(placeholder_modelo)
                | serie_pred.astype(str).eq(placeholder_modelo)
            )
        else:
            mascara_placeholder = pd.Series(False, index=df_pred.index)

        clas_num = pd.to_numeric(serie_clas, errors='coerce')
        pred_num = pd.to_numeric(serie_pred, errors='coerce')

        serie_final = pred_num.where(clas_num <= 0, -1).round(2)
        serie_final = serie_final.astype(object)
        serie_final[mascara_placeholder | clas_num.isna() | pred_num.isna()] = placeholder_modelo
        df_pred[col_final] = _normalizar_columna_mixta(serie_final, placeholder=placeholder_modelo)
        return df_pred

    for asig in asig_a_usar:
        try:
            print(\"[Resultados Funcion : INFO] \", '\\n' + '-' * 70)
            print(\"[Resultados Funcion : INFO] \", 'INICIANDO PROCESO POR ASIGNATURA')
            print(\"[Resultados Funcion : INFO] \", '-' * 70)

            df_resultado_asig = df_usar[df_usar['Cod materia curso'] == asig].copy()
            nombre_asig = df_resultado_asig['Descripcion_Materia'].iloc[0]
            print(\"[Resultados Funcion : INFO] \", f'\\n == Resultados para programa:  {asig} - {nombre_asig} == \\n')

            df_usar_filtrado = df_resultado_asig.copy()
            tipo_asig_prereq = df_usar_filtrado['Observacion_Prerrequisito'].iloc[0]
            print(\"[Resultados Funcion : INFO] \", '-------------- Preparando columnas / prerequisitos --------------')

            col_usar = []
            var_objetivo = ''
            lista_prereq_usar = []
            var_objetivo_clas = 'Retiro_Asignatura_Cat'

            if tipo_asig_prereq == 'Prerrequisito cumplido':
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(df_usar_filtrado, tiene_prereq=True)
                lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext(
                    df_usar_filtrado, df_historial_asignaturas_nombres, 0.8
                )
                print(\"[Resultados Funcion : INFO] \", '[Info] Asignatura con prerequisitos. Usando logica CON PRE REQUISITOS.')
            else:
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(df_usar_filtrado, tiene_prereq=False)
                lista_prereq_usar = columnas_prereq_validas(df_usar_filtrado, 0.8)
                print(\"[Resultados Funcion : INFO] \", '[Info] Asignatura sin prerequisitos. Usando logica SIN PRE REQUISITOS.')

            col_usar = col_usar + lista_prereq_usar
            print(\"[Resultados Funcion : INFO] \", f'Columnas a usar ({len(col_usar)}): {col_usar} \\n  Numero de filas a tener en cuenta: {len(df_usar_filtrado)}')

            cols_to_category = [
                'programa',
                'sexo',
                'procedencia_categoria',
                'profesor_codigo',
                'Tipo_colegio',
                'Tipo_calendario'
            ]

            df_usar_filtrado, col_usar, faltantes = preparar_dataframe_para_modelado(
                df_usar_filtrado,
                col_usar,
                var_objetivo,
                var_objetivo_clas,
            )

            num_filas_eliminadas = 0
            print(\"[Resultados Funcion : INFO] \", '[Info] En inferencia con modelos guardados no se eliminan filas por nulos; se conservan todas para alinear y predecir.')
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)
            print(\"[Resultados Funcion : INFO] \", '-------------- Preprocesamiento completado --------------')

            df_base = df_usar_filtrado.copy()

            cols_xgb = [
                'Clasificacion_XGB',
                'Prediccion_XGB',
                'Prediccion_final_XGB',
                'interpretacion_general_xgb',
                'interpretacion_registro_xgb',
            ]
            cols_cat = [
                'Clasificacion_CAT',
                'Prediccion_CAT',
                'Prediccion_final_CAT',
                'interpretacion_general_cat',
                'interpretacion_registro_cat',
            ]

            df_pred_xgb = df_base.copy()
            df_pred_cat = df_base.copy()

            # =========================
            # XGBOOST (clasificacion + regresion)
            # =========================
            try:
                print(\"[Resultados Funcion : INFO] \", '-------------- Corriendo modelo XGBoost --------------')
                modelo_clasif_xgb = cargar_modelo_xg_clasificacion(ruta_modelos_cls_xgb, asig, variant=variant_modelo)
                modelo_reg_xgb = cargar_modelo_xg_regresion(ruta_modelos_reg_xgb, asig, variant=variant_modelo)

                if modelo_clasif_xgb is None:
                    print(\"[Resultados Funcion : INFO] \", '[Aviso] Esta asignatura no tiene modelo de clasificacion XGBoost.')
                    df_pred_xgb = _marcar_no_disponible(df_pred_xgb, ['Clasificacion_XGB'])
                else:
                    features_xgb_cls = _resolver_features(
                        asig,
                        'xgboost',
                        'prediccion_retiro',
                        modelo_clasif_xgb,
                        df_base.columns.tolist(),
                        excluir=[var_objetivo, var_objetivo_clas]
                    )
                    print(\"[Resultados Funcion : INFO] \", f'Las columnas esperadas (XGB-CLS) son {features_xgb_cls}')
                    df_xgb_cls = alinear_dataframe_a_modelo(df_base, features_xgb_cls, fill_value=0.0)
                    df_clasif_xgb = usar_xgboost_clasificacion(
                        df=df_xgb_cls,
                        modelo=modelo_clasif_xgb,
                        columna_objetivo=var_objetivo_clas,
                        columnas_predictores=features_xgb_cls
                    )
                    df_pred_xgb['Clasificacion_XGB'] = df_clasif_xgb['Clasificacion_XGB']

                if modelo_reg_xgb is None:
                    print(\"[Resultados Funcion : INFO] \", '[Aviso] Esta asignatura no tiene modelo de regresion XGBoost.')
                    df_pred_xgb = _marcar_no_disponible(
                        df_pred_xgb,
                        ['Prediccion_XGB', 'Prediccion_final_XGB', 'interpretacion_general_xgb', 'interpretacion_registro_xgb']
                    )
                else:
                    features_xgb_reg = _resolver_features(
                        asig,
                        'xgboost',
                        'prediccion_nota',
                        modelo_reg_xgb,
                        df_base.columns.tolist(),
                        excluir=[var_objetivo, var_objetivo_clas]
                    )
                    print(\"[Resultados Funcion : INFO] \", f'Las columnas esperadas (XGB-REG) son {features_xgb_reg}')
                    df_xgb_reg = alinear_dataframe_a_modelo(df_base, features_xgb_reg, fill_value=0.0)
                    df_pred_xgb_reg = usar_xgboost_regresion(
                        df=df_xgb_reg,
                        modelo=modelo_reg_xgb,
                        columna_objetivo=var_objetivo,
                        columnas_predictores=features_xgb_reg
                    )
                    df_pred_xgb['Prediccion_XGB'] = df_pred_xgb_reg['Prediccion_XGB']
                    df_pred_xgb = cambiar_a_category(df_pred_xgb, cols_to_category + ['interpretacion_general_xgb', 'interpretacion_registro_xgb'])
                    print(\"[Resultados Funcion : INFO] \", '[Interpretacion] Insertando interpretaciones SHAP (XGB) en el dataframe:')
                    df_pred_xgb, _ = escribir_interpretaciones_shap_modelo_cargado(
                        modelo_reg_xgb,
                        df_xgb_reg[features_xgb_reg],
                        df_pred_xgb,
                        top_n=10,
                        col_general='interpretacion_general_xgb',
                        col_registro='interpretacion_registro_xgb'
                    )
                    df_pred_xgb = _clip_predicciones(df_pred_xgb, 'Prediccion_XGB')
                    df_pred_xgb['Prediccion_XGB'] = _normalizar_columna_mixta(df_pred_xgb['Prediccion_XGB'], placeholder=placeholder_modelo)

                df_pred_xgb = _calcular_prediccion_final(
                    df_pred_xgb,
                    'Clasificacion_XGB',
                    'Prediccion_XGB',
                    'Prediccion_final_XGB'
                )
                print(\"[Resultados Funcion : INFO] \", '-------------- Modelo XGBoost ejecutado con exito --------------')
            except Exception as e:
                print(\"[Resultados Funcion : INFO] \", f'[Aviso] Error XGB en {asig}: {e}')
                df_pred_xgb = _marcar_no_disponible(df_base.copy(), cols_xgb)

            # =========================
            # CATBOOST (clasificacion + regresion)
            # =========================
            try:
                print(\"[Resultados Funcion : INFO] \", '-------------- Corriendo modelo CatBoost --------------')
                modelo_clasif_cat = cargar_modelo_catboost_clasificacion(ruta_modelos_cls_cat, asig, variant=variant_modelo)
                modelo_reg_cat = cargar_modelo_catboost_regresion(ruta_modelos_reg_cat, asig, variant=variant_modelo)

                if modelo_clasif_cat is None:
                    print(\"[Resultados Funcion : INFO] \", '[Aviso] Esta asignatura no tiene modelo de clasificacion CatBoost.')
                    df_pred_cat = _marcar_no_disponible(df_pred_cat, ['Clasificacion_CAT'])
                else:
                    features_cat_cls = _resolver_features(
                        asig,
                        'catboost',
                        'prediccion_retiro',
                        modelo_clasif_cat,
                        df_base.columns.tolist(),
                        excluir=[var_objetivo, var_objetivo_clas]
                    )
                    print(\"[Resultados Funcion : INFO] \", f'Las columnas esperadas (CAT-CLS) son {features_cat_cls}')
                    df_cat_cls = alinear_dataframe_a_modelo(df_base, features_cat_cls, fill_value=np.nan)
                    X_clasif, _ = preparar_X_numerico(df_cat_cls, features_cat_cls)
                    y_pred_cls = np.array(modelo_clasif_cat.predict(X_clasif)).astype(int).ravel()
                    df_pred_cat['Clasificacion_CAT'] = y_pred_cls

                if modelo_reg_cat is None:
                    print(\"[Resultados Funcion : INFO] \", '[Aviso] Esta asignatura no tiene modelo de regresion CatBoost.')
                    df_pred_cat = _marcar_no_disponible(
                        df_pred_cat,
                        ['Prediccion_CAT', 'Prediccion_final_CAT', 'interpretacion_general_cat', 'interpretacion_registro_cat']
                    )
                else:
                    features_cat_reg = _resolver_features(
                        asig,
                        'catboost',
                        'prediccion_nota',
                        modelo_reg_cat,
                        df_base.columns.tolist(),
                        excluir=[var_objetivo, var_objetivo_clas]
                    )
                    print(\"[Resultados Funcion : INFO] \", f'Las columnas esperadas (CAT-REG) son {features_cat_reg}')
                    df_cat_reg = alinear_dataframe_a_modelo(df_base, features_cat_reg, fill_value=np.nan)
                    X_reg, _ = preparar_X_numerico(df_cat_reg, features_cat_reg)
                    y_pred_reg = modelo_reg_cat.predict(X_reg)
                    df_pred_cat['Prediccion_CAT'] = y_pred_reg
                    df_pred_cat = cambiar_a_category(df_pred_cat, cols_to_category + ['interpretacion_general_cat', 'interpretacion_registro_cat'])
                    print(\"[Resultados Funcion : INFO] \", '[Interpretacion] Insertando interpretaciones SHAP (CAT) en el dataframe:')
                    df_pred_cat, _ = escribir_interpretaciones_shap_catboost_modelo_cargado(
                        modelo_reg_cat,
                        df_cat_reg[features_cat_reg],
                        df_pred_cat,
                        top_n=10,
                        col_general='interpretacion_general_cat',
                        col_registro='interpretacion_registro_cat'
                    )
                    df_pred_cat = _clip_predicciones(df_pred_cat, 'Prediccion_CAT')
                    df_pred_cat['Prediccion_CAT'] = _normalizar_columna_mixta(df_pred_cat['Prediccion_CAT'], placeholder=placeholder_modelo)

                df_pred_cat = _calcular_prediccion_final(
                    df_pred_cat,
                    'Clasificacion_CAT',
                    'Prediccion_CAT',
                    'Prediccion_final_CAT'
                )
                df_pred_cat_ultimo = df_pred_cat.copy()
                print(\"[Resultados Funcion : INFO] \", '-------------- Modelo CatBoost ejecutado con exito --------------')
            except Exception as e:
                print(\"[Resultados Funcion : INFO] \", f'[Aviso] Error CAT en {asig}: {e}')
                df_pred_cat = _marcar_no_disponible(df_base.copy(), cols_cat)

            df_resultado_asig = _inyectar_columnas(df_resultado_asig, df_pred_xgb, cols_xgb)
            df_resultado_asig = _inyectar_columnas(df_resultado_asig, df_pred_cat, cols_cat)

            cols_to_category = cols_to_category + [
                'interpretacion_general_xgb',
                'interpretacion_registro_xgb',
                'interpretacion_general_cat',
                'interpretacion_registro_cat',
            ]
            df_resultado_asig = cambiar_a_category(df_resultado_asig, cols_to_category)

            if df_resultados_final is None or df_resultados_final.empty:
                df_resultados_final = df_resultado_asig.copy()
            else:
                df_resultados_final = pd.concat([df_resultados_final, df_resultado_asig], axis=0, sort=False)

            print(\"[Resultados Funcion : INFO] \", '[OK] Asignatura procesada y agregada a df_resultados_final')

        except Exception as e:
            print(\"[Resultados Funcion : INFO] \", f'Error al procesar la asignatura {asig}: {e}')
            if df_errores.empty:
                df_errores = pd.DataFrame()
            try:
                nombre_asig = df_usar[df_usar['Cod materia curso'] == asig]['Descripcion_Materia'].iloc[0]
            except Exception:
                nombre_asig = ''
            fila_error = pd.DataFrame([
                {'Cod materia curso': asig, 'Descripcion_Materia': nombre_asig, 'Error': f'{type(e).__name__}: {e}'}
            ])
            df_errores = pd.concat([df_errores, fila_error], ignore_index=True, sort=False)

    df_resultados_final = _reordenar_prereq_al_final(df_resultados_final)

    carpeta_resultados_final = str(NOTEBOOK_DIR / f'Resultados_Ejecucion_Modelo_{MODEL_VERSION}') + os.sep
    guardar_resultados(
        df_resultados_final,
        carpeta_resultados_final,
        f'df_resultados_modelos_guardados_{variant_modelo}_'
    )

    return df_resultados_final, df_errores, df_pred_cat_ultimo
"""


PREP_CELL = """#Limpiar y cargar DataFrame para correr modelos por asignatura usando modelos guardados
## Preparar datos para usar guardados modelos por asignatura



periodo_a_evaluar=202530
ruta_modelos_guardados = get_model_dir('xgboost', 'main', 'prediccion_nota')
asignaturas_mod_guardados=check_asignaturas_en_carpeta(ruta_modelos_guardados)

cols_to_excl =[
        \"Nombre_Programa\",
        \"_ Matricula detalle para analisis.Prof_Codigo\",
        \"_ Matricula detalle para analisis.Sexo\",
        \"_ Matricula detalle para analisis.Procedencia Categoria\",
    ]

#[\"FIS1023\",\"MAT1111\",\"FIS1033\"]  Prueba donde tenia errores antes -> # ['FRA1010','MAT1121','IST4360','PSI1160']  # ['CMN7190'] 
asig_a_usar= asignaturas_mod_guardados#[\"FIS1023\"]#,\"MAT1111\",\"FIS1033\"]#[\"FIS1023\",\"MAT1111\",\"EST7042\",\"IST2089\",\"MAT4011\",\"IBA4032\",\"MAT4258\",\"MAT4260\",\"FIS1033\",\"FIS1043\"]
#asig_a_usar = [\"FIS1023\",\"MAT1111\",\"FIS1033\"]  #asignaturas_mod_guardados[:10]
#asig_a_usar=[\"ECO2011\"]

df_usar = df_historial[
    ((df_historial[\"Observacion_Prerrequisito\"] == \"Prerrequisito cumplido\") | (df_historial[\"Observacion_Prerrequisito\"] == \"No tiene pre requisito\")
      ) &
    (df_historial[\"Cod materia curso\"].isin(asig_a_usar) )
].copy()
df_usar=arreglar_comas_por_puntos(df_usar,cols_to_excl)
df_usar=limpiar_dataframe(df_usar, True)

if \"Periodo\" in df_usar.columns:
    df_usar = df_usar[df_usar[\"Periodo\"]==periodo_a_evaluar]
else:
    print(\"[Resultados Funcion : INFO] \", \"[Aviso] La columna 'Periodo' no existe despues de limpiar_dataframe; se omite el filtro por periodo.\")

col_dpto_src = \"_ Matricula detalle para analisis.DPTO Asignatura\"
if col_dpto_src in df_usar.columns:
    df_usar[\"DPTO Asignatura\"]=df_usar[col_dpto_src]
"""

USAR_XGB_REG_FUNC = """def usar_xgboost_regresion(df, modelo ,columna_objetivo, columnas_predictores,nombre_col_pred="Prediccion_XGB"):


    # Separar variables
    X = df[columnas_predictores].copy()

    # Agregar columna de predicciones al df completo
    df_resultados = df.copy()
    df_resultados[nombre_col_pred] = modelo.predict(X)

    return df_resultados
"""

INTERP_XGB_CARGADO_FUNC = """def escribir_interpretaciones_shap_modelo_cargado(modelo,X,df,top_n=10,col_general="interpretacion_general",col_registro="interpretacion_registro"):
    \"\"\"
    Escribe interpretaciones SHAP para un modelo XGBoost ya cargado.
    \"\"\"

    import numpy as np
    import pandas as pd
    import shap

    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    usa_categoricas = any(pd.api.types.is_categorical_dtype(dt) for dt in X.dtypes)

    if usa_categoricas:
        dmx = xgb.DMatrix(X, enable_categorical=True)
        explainer = shap.TreeExplainer(modelo.get_booster())
        shap_exp = explainer(dmx)
    else:
        explainer = shap.TreeExplainer(modelo)
        shap_exp = explainer(X)

    vals = shap_exp.values
    importancia_media = pd.Series(np.abs(vals).mean(axis=0), index=X.columns).sort_values(ascending=False)

    top_imp = importancia_media.head(top_n)
    interp_general = "Variables con mayor impacto promedio (|SHAP|): " + ", ".join(
        f"{feat} ({imp:.3f})" for feat, imp in top_imp.items()
    )
    general_series = pd.Series(interp_general, index=X.index)
    print("[Resultados Funcion : INFO] ", "[Info] Interpretacion general generada.")
    print("[Resultados Funcion : INFO] ", interp_general)

    def resumen_por_fila(row_vals):
        contrib = pd.Series(row_vals, index=X.columns)
        pos = contrib[contrib > 0].abs().sort_values(ascending=False).head(top_n)
        neg = contrib[contrib < 0].abs().sort_values(ascending=False).head(top_n)
        pos_str = ", ".join(f"{f} (+{contrib[f]:.3f})" for f in pos.index) if len(pos) else "0"
        neg_str = ", ".join(f"{f} ({contrib[f]:.3f})" for f in neg.index) if len(neg) else "0"
        return f"A favor: {pos_str} | En contra: {neg_str}"

    print("[Resultados Funcion : INFO] ", "[Info] Generando resumen por fila...")
    interp_registro_series = pd.Series(
        (resumen_por_fila(vals[i, :]) for i in range(vals.shape[0])),
        index=X.index
    )

    if len(df) == len(X) and df.index.equals(X.index):
        df[col_general] = general_series
        df[col_registro] = interp_registro_series
    else:
        try:
            df.loc[X.index, col_general] = general_series
            df.loc[X.index, col_registro] = interp_registro_series
        except Exception:
            if len(df) == len(X):
                df[col_general] = general_series.values
                df[col_registro] = interp_registro_series.values
            else:
                raise ValueError("No coinciden longitudes/indices de df y X; no se pueden escribir interpretaciones de forma segura.")

    return df, importancia_media
"""


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    tmp_lines = Path("_tmp_rev_multiples_asig_rev_ia.py").read_text(encoding="utf-8").splitlines()
    cell5 = "\n".join(tmp_lines[26:929]) + "\n"
    old_check_start = cell5.index("## check asgianturas en carpeta")
    old_check_end = cell5.index("def elegir_asignaturas(")
    cell5 = cell5[:old_check_start] + CHECK_FUNC + "\n\n" + cell5[old_check_end:]
    cell5 = cell5.replace("    return X, cat_cols\n", "    return X, cat_cols\n\n" + PREPARAR_HELPERS + "\n")

    nb["cells"][5]["source"] = cell5.splitlines(keepends=True)
    nb["cells"][15]["source"] = [line + "\n" for line in USAR_MODELOS_FUNC.splitlines()]
    nb["cells"][28]["source"] = [line + "\n" for line in PREP_CELL.splitlines()]

    cell11 = "".join(nb["cells"][11]["source"])
    cell11 = cell11.replace(
        "df.to_csv(nombre_archivo, index=False, sep=';', float_format='%.3f', decimal=',')",
        "df.to_csv(nombre_archivo, index=False, sep=';', float_format='%.3f', decimal=',', encoding='utf-8-sig')"
    )
    inicio = cell11.index('def usar_xgboost_regresion(')
    fin = cell11.index('def usar_xgboost_clasificacion(')
    cell11 = cell11[:inicio] + USAR_XGB_REG_FUNC + "\n\n" + cell11[fin:]
    if 'def escribir_interpretaciones_shap_modelo_cargado(' not in cell11:
        marker = 'def alinear_dataframe_a_modelo('
        cell11 = cell11.replace(marker, INTERP_XGB_CARGADO_FUNC + "\n\n" + marker)
    nb["cells"][11]["source"] = cell11.splitlines(keepends=True)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook actualizado.")


if __name__ == "__main__":
    main()
