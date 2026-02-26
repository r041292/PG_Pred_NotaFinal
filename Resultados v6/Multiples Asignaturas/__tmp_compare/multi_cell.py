# Mega pipeline con XGBoost, RF y CatBoost por asignatura
import os

def correr_modelos_multi_por_asignatura(df_usar):
    """
    Ejecuta pipeline completo por asignatura con XGBoost, RandomForest y CatBoost
    (clasificacion de retiro y regresion de nota) manteniendo la misma logica
    de seleccion de asignaturas, presentacion de resultados y determinacion
    de variables.

    Retorna:
        df_resultados_final: DataFrame con datos originales y columnas de prediccion
        df_resultados_stats: Metricas de regresion por asignatura y modelo
        df_resultados_class_retiro_stats: Metricas de clasificacion por asignatura y modelo
    """

    col_usar = []
    var_objetivo = ''

    df_resultados_final = pd.DataFrame()
    df_resultados_stats = pd.DataFrame()
    df_resultados_class_retiro_stats = pd.DataFrame()

    asig_a_usar = df_usar['Cod materia curso'].unique().tolist()

    # Rutas para guardar modelos
    ruta_modelos_reg_xgb = get_model_dir('xgboost', 'main', 'prediccion_nota')
    ruta_modelos_cls_xgb = get_model_dir('xgboost', 'main', 'prediccion_retiro')
    ruta_modelos_reg_rf = get_model_dir('rf', 'main', 'prediccion_nota')
    ruta_modelos_cls_rf = get_model_dir('rf', 'main', 'prediccion_retiro')
    ruta_modelos_reg_cat = get_model_dir('catboost', 'main', 'prediccion_nota')
    ruta_modelos_cls_cat = get_model_dir('catboost', 'main', 'prediccion_retiro')

    for p in [ruta_modelos_reg_xgb, ruta_modelos_cls_xgb, ruta_modelos_reg_rf, ruta_modelos_cls_rf, ruta_modelos_reg_cat, ruta_modelos_cls_cat]:
        os.makedirs(p, exist_ok=True)

    for asig in asig_a_usar:
        try:
            print('\n' + '=' * 80)
            nombre_asig = df_usar[df_usar['Cod materia curso'] == asig]['Descripcion_Materia'].iloc[0]
            print(f'\n == Resultados para programa: {asig} - {nombre_asig} == \n')

            df_usar_filtrado = df_usar[df_usar['Cod materia curso'] == asig].copy()

            tipo_asig = None
            if 'Observacion_Prerrequisito' in df_usar_filtrado.columns:
                tipo_series = df_usar_filtrado['Observacion_Prerrequisito'].dropna()
                if len(tipo_series) > 0:
                    tipo_asig = tipo_series.iloc[0]

            if tipo_asig == 'Prerrequisito cumplido':
                print('[Info] Asignatura con prerequisitos. Usando logica CON PRE REQUISITOS.')
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado, tiene_prereq=True
                )
                lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext_dif_prereq(
                    df_usar_filtrado, df_historial_asignaturas_nombres, 0.8
                )
            else:
                print('[Info] Asignatura sin prerequisitos. Usando logica SIN PRE REQUISITOS.')
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado, tiene_prereq=False
                )
                lista_prereq_usar = columnas_prereq_validas(df_usar_filtrado, 0.8)

            col_usar = col_usar + lista_prereq_usar

            print(f'Columnas a usar ({len(col_usar)}): {col_usar} \n  Numero de filas a tener en cuenta: {len(df_usar_filtrado)}')

            cols_to_category = [
                'programa',
                'sexo',
                'procedencia_categoria',
                'profesor_codigo',
                'Tipo_colegio',
                'Tipo_calendario',
            ]

            var_objetivo_clas = 'Retiro_Asignatura_Cat'
            
            if var_objetivo_clas in df_usar_filtrado.columns:
                df_usar_filtrado[var_objetivo_clas] = df_usar_filtrado[var_objetivo_clas].astype(int)

            cols_present = [c for c in col_usar if c in df_usar_filtrado.columns]
            if var_objetivo in df_usar_filtrado.columns and var_objetivo not in cols_present:
                cols_present.append(var_objetivo)
            if var_objetivo_clas in df_usar_filtrado.columns and var_objetivo_clas not in cols_present:
                cols_present.append(var_objetivo_clas)

            faltantes = [c for c in col_usar if c not in df_usar_filtrado.columns]
            if faltantes:
                print(f'[Aviso] Algunas columnas de col_usar no existen y se omiten: {faltantes}')

            df_usar_filtrado = df_usar_filtrado[cols_present].copy()
            print(f'[Info] df_usar_filtrado reducido a {df_usar_filtrado.shape[1]} columnas y {len(df_usar_filtrado)} filas')

            df_usar_filtrado, num_filas_eliminadas = eliminar_filas_por_columna(df_usar_filtrado)
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            col_usar = [elemento for elemento in col_usar if elemento not in faltantes]

            num_estud_retiros = len(df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 1])
            num_total_estud = len(df_usar_filtrado)
            porc_retiros = num_estud_retiros / num_total_estud

            print(f'[Aviso] El porcentaje de retiros es  ({porc_retiros:.2%}).')

            if porc_retiros > 0.03:
                # ==============================
                # Clasificacion de retiros (XGB)
                # ==============================
                print('\n-------------------Modelo de Clasificacion de Retiros (XGB)--------------------\n')
                modelo_clasif_xgb, df_pred_clasif_xgb, metricas_clasif_xgb, X_train_clasif_xgb, X_test_clasif_xgb, y_train_clasif_xgb, y_test_clasif_xgb, df_resultados_stats_class_temp_xgb = entrenar_xgboost_classif(
                    df=df_usar_filtrado,
                    columna_objetivo=var_objetivo_clas,
                    columnas_predictores=col_usar
                )
                col_prediccion_clasif_xgb = 'Clasificacion_XGB'
                df_usar_filtrado[col_prediccion_clasif_xgb] = df_pred_clasif_xgb[col_prediccion_clasif_xgb]
                df_resultados_stats_class_temp_xgb['Descripcion_Materia'] = nombre_asig
                df_resultados_stats_class_temp_xgb['Cod materia curso'] = asig
                df_resultados_stats_class_temp_xgb['Modelo'] = 'XGB'
                df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp_xgb], axis=0)
                guardar_xgboost_modelo(modelo_clasif_xgb, ruta_modelos_clasificacion=ruta_modelos_cls_xgb, cod_asig=asig)

                # ==============================
                # Clasificacion de retiros (RF)
                # ==============================
                print('\n-------------------Modelo de Clasificacion de Retiros (RF)--------------------\n')
                modelo_clasif_rf, df_pred_clasif_rf, metricas_clasif_rf, X_train_clasif_rf, X_test_clasif_rf, y_train_clasif_rf, y_test_clasif_rf, df_resultados_stats_class_temp_rf = entrenar_rf_classif(
                    df=df_usar_filtrado,
                    columna_objetivo=var_objetivo_clas,
                    columnas_predictores=col_usar
                )
                col_prediccion_clasif_rf = 'Clasificacion_RF'
                df_usar_filtrado[col_prediccion_clasif_rf] = df_pred_clasif_rf[col_prediccion_clasif_rf]
                df_resultados_stats_class_temp_rf['Descripcion_Materia'] = nombre_asig
                df_resultados_stats_class_temp_rf['Cod materia curso'] = asig
                df_resultados_stats_class_temp_rf['Modelo'] = 'RF'
                df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp_rf], axis=0)
                guardar_sklearn_modelo(modelo_clasif_rf, ruta_modelos_clasificacion=ruta_modelos_cls_rf, cod_asig=asig, prefijo='rf')

                # ==============================
                # Clasificacion de retiros (CAT)
                # ==============================
                print('\n-------------------Modelo de Clasificacion de Retiros (CAT)--------------------\n')
                modelo_clasif_cat, df_pred_clasif_cat, metricas_clasif_cat, X_train_clasif_cat, X_test_clasif_cat, y_train_clasif_cat, y_test_clasif_cat, df_resultados_stats_class_temp_cat = entrenar_catboost_classif(
                    df=df_usar_filtrado,
                    columna_objetivo=var_objetivo_clas,
                    columnas_predictores=col_usar
                )
                col_prediccion_clasif_cat = 'Clasificacion_CAT'
                df_usar_filtrado[col_prediccion_clasif_cat] = df_pred_clasif_cat[col_prediccion_clasif_cat]
                df_resultados_stats_class_temp_cat['Descripcion_Materia'] = nombre_asig
                df_resultados_stats_class_temp_cat['Cod materia curso'] = asig
                df_resultados_stats_class_temp_cat['Modelo'] = 'CAT'
                df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp_cat], axis=0)
                guardar_catboost_modelo(modelo_clasif_cat, ruta_modelos_clasificacion=ruta_modelos_cls_cat, cod_asig=asig)

            # ==============================
            # Regresion de nota (XGB)
            # ==============================
            print('\n-------------------Modelo de Regresion de Nota (XGB)--------------------\n')
            modelo_xgb, df_pred_xgb, metricas_xgb, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, df_resultados_stats_temp_xgb = entrenar_xgboost_regresion(
                df=df_usar_filtrado,
                columna_objetivo=var_objetivo,
                columnas_predictores=col_usar,
                tuning=False
            )
            col_prediccion_xgb = 'Prediccion_XGB'
            df_usar_filtrado[col_prediccion_xgb] = df_pred_xgb[col_prediccion_xgb]
            df_resultados_stats_temp_xgb['Descripcion_Materia'] = nombre_asig
            df_resultados_stats_temp_xgb['Cod materia curso'] = asig
            df_resultados_stats_temp_xgb['Modelo'] = 'XGB'
            df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp_xgb], axis=0)
            guardar_xgboost_modelo(modelo_xgb, ruta_modelos_regresion=ruta_modelos_reg_xgb, cod_asig=asig)
            interpretar_xgboost_shap(modelo_xgb, X_test_xgb, col_usar, top_n=1, id_check=0)


            # ==============================
            # 5) INTERPRETACIONES SHAP EN EL DATAFRAME DE PREDICCIONES (XGB)
            # ==============================
            cols_interpret_xgb = ["interpretacion_general_xgb", "interpretacion_registro_xgb"]
            df_pred_xgb = cambiar_a_category(df_pred_xgb, cols_to_category + cols_interpret_xgb)
            print("[Aviso] Insertando interpretaciones SHAP (XGB) en el dataframe de predicciones:")
            df_pred_xgb, importancia_media_xgb = escribir_interpretaciones_shap_xgb(
                modelo_xgb,
                df_pred_xgb[col_usar],
                df_pred_xgb,
                top_n=10,
                col_general=cols_interpret_xgb[0],
                col_registro=cols_interpret_xgb[1],
            )
            df_to_add = df_pred_xgb[cols_interpret_xgb].reindex(df_usar_filtrado.index)
            df_usar_filtrado = df_usar_filtrado.join(df_to_add)

            # ==============================
            # Regresion de nota (RF)
            # ==============================
            
            print('\n-------------------Modelo de Regresion de Nota (RF)--------------------\n')
            modelo_rf, df_pred_rf, metricas_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf, df_resultados_stats_temp_rf = entrenar_rf_regresion(
                df=df_usar_filtrado,
                columna_objetivo=var_objetivo,
                columnas_predictores=col_usar,
                tuning=False
            )
            col_prediccion_rf = 'Prediccion_RF'
            df_usar_filtrado[col_prediccion_rf] = df_pred_rf[col_prediccion_rf]
            df_resultados_stats_temp_rf['Descripcion_Materia'] = nombre_asig
            df_resultados_stats_temp_rf['Cod materia curso'] = asig
            df_resultados_stats_temp_rf['Modelo'] = 'RF'
            df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp_rf], axis=0)
            guardar_sklearn_modelo(modelo_rf, ruta_modelos_regresion=ruta_modelos_reg_rf, cod_asig=asig, prefijo='rf')
            interpretar_rf_shap(modelo_rf, X_test_rf, col_usar, top_n=1, id_check=0)


            # ==============================
            # 5) INTERPRETACIONES SHAP EN EL DATAFRAME DE PREDICCIONES (RF)
            # ==============================
            '''
            cols_interpret_rf = ["interpretacion_general_rf", "interpretacion_registro_rf"]
            df_pred_rf = cambiar_a_category(df_pred_rf, cols_to_category + cols_interpret_rf)
            print("[Aviso] Insertando interpretaciones SHAP (RF) en el dataframe de predicciones:")
            df_pred_rf, importancia_media_rf = escribir_interpretaciones_shap_rf(
                modelo_rf,
                df_pred_rf[col_usar],
                df_pred_rf,
                top_n=10,
                col_general=cols_interpret_rf[0],
                col_registro=cols_interpret_rf[1],
            )
            df_to_add = df_pred_rf[cols_interpret_rf].reindex(df_usar_filtrado.index)
            df_usar_filtrado = df_usar_filtrado.join(df_to_add)
            '''

            # ==============================
            # Regresion de nota (CAT)
            # ==============================
            print('\n-------------------Modelo de Regresion de Nota (CAT)--------------------\n')
            modelo_cat, df_pred_cat, metricas_cat, X_train_cat, X_test_cat, y_train_cat, y_test_cat, df_resultados_stats_temp_cat = entrenar_catboost_regresion(
                df=df_usar_filtrado,
                columna_objetivo=var_objetivo,
                columnas_predictores=col_usar,
                tuning=False
            )
            col_prediccion_cat = 'Prediccion_CAT'
            df_usar_filtrado[col_prediccion_cat] = df_pred_cat[col_prediccion_cat]
            df_resultados_stats_temp_cat['Descripcion_Materia'] = nombre_asig
            df_resultados_stats_temp_cat['Cod materia curso'] = asig
            df_resultados_stats_temp_cat['Modelo'] = 'CAT'
            df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp_cat], axis=0)
            guardar_catboost_modelo(modelo_cat, ruta_modelos_regresion=ruta_modelos_reg_cat, cod_asig=asig)
            interpretar_catboost_shap(modelo_cat, X_test_cat, col_usar, top_n=1, id_check=0)


            # ==============================
            # 5) INTERPRETACIONES SHAP EN EL DATAFRAME DE PREDICCIONES (CAT)
            # ==============================
            cols_interpret_cat = ["interpretacion_general_cat", "interpretacion_registro_cat"]
            df_pred_cat = cambiar_a_category(df_pred_cat, cols_to_category + cols_interpret_cat)
            print("[Aviso] Insertando interpretaciones SHAP (CAT) en el dataframe de predicciones:")
            df_pred_cat, importancia_media_cat = escribir_interpretaciones_shap_catboost(
                modelo_cat,
                df_pred_cat[col_usar],
                df_pred_cat,
                top_n=10,
                col_general=cols_interpret_cat[0],
                col_registro=cols_interpret_cat[1],
            )
            df_to_add = df_pred_cat[cols_interpret_cat].reindex(df_usar_filtrado.index)
            df_usar_filtrado = df_usar_filtrado.join(df_to_add)
            

            # Añadir identificadores de asignatura
            df_usar_filtrado["Descripcion_Materia"] = nombre_asig
            df_usar_filtrado["Cod materia curso"] = asig

            # Añadir variables adicionales desde df_usar original (si existen)
            cols_extra = ["Periodo", "Nombre_Division", "cohorte", "Pidm", "nrc", "DPTO Asignatura"]

            cols_extra_presentes = [c for c in cols_extra if c in df_usar.columns]
            if cols_extra_presentes:
                extra_info = (
                    df_usar[df_usar["Cod materia curso"] == asig][cols_extra_presentes]
                    .reindex(df_usar_filtrado.index)
                )
                df_usar_filtrado = df_usar_filtrado.join(extra_info)

            cols_to_category = cols_to_category + [
                "interpretacion_general_xgb",
                "interpretacion_registro_xgb",
                "interpretacion_general_rf",
                "interpretacion_registro_rf",
                "interpretacion_general_cat",
                "interpretacion_registro_cat",
            ]
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            if df_resultados_final is None or df_resultados_final.empty:
                df_resultados_final = df_usar_filtrado
            else:
                df_resultados_final = pd.concat([df_resultados_final, df_usar_filtrado], axis=0)

        except Exception as e:
            print(f'[Error] Fallo procesando asignatura {asig}: {e}')

            try:
                nombre_asig = df_usar[df_usar["Cod materia curso"] == asig]["Descripcion_Materia"].iloc[0]
            except Exception:
                nombre_asig = str(asig)

            error_row = pd.DataFrame(
                [
                    {
                        "mean": "error",
                        "std": "error",
                        "metric": f"{type(e).__name__}: {e}",
                        "Descripcion_Materia": nombre_asig,
                        "Cod materia curso": asig,
                    }
                ]
            )

            if not isinstance(df_resultados_stats, pd.DataFrame):
                try:
                    df_resultados_stats = pd.DataFrame(df_resultados_stats)
                except Exception:
                    df_resultados_stats = pd.DataFrame()

            df_resultados_stats = pd.concat(
                [df_resultados_stats, error_row],
                ignore_index=True,
                sort=False,
            )

            # Continúa con la siguiente asignatura
            continue

    return df_resultados_final, df_resultados_stats, df_resultados_class_retiro_stats
