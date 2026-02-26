def correr_modelos_xgboost_por_asignatura(df_usar):
    """
    Ejecuta el pipeline completo de XGBoost (clasificación de retiro y regresión de nota)
    por asignatura, seleccionando automáticamente la lógica de preprocesamiento según
    la columna `tipo_asig_prereq`.

    Parámetros
    ----------
    df_usar : pd.DataFrame
        DataFrame con la información de los estudiantes, que incluye al menos:
        - "Cod materia curso"
        - "Descripcion_Materia"
        - "tipo_asig_prereq" (para decidir la lógica de prerequisitos)
        - Todas las columnas necesarias por las funciones auxiliares ya definidas:
          renombrar_columnas, columnas_prereq_validas, columnas_prereq_validas_ext, etc.

    Lógica de selección
    -------------------
    Para cada asignatura:
    - Si df_usar_filtrado["tipo_asig_prereq"] == "Prerrequisito cumplido":
        * Usa la lógica equivalente a:
          ## CORRER XGBOOST POR ASIGNATURA CON PRE REQUISITOS
        * Llama a:
          df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(..., tiene_prereq=True)
          lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext(...)
    - En cualquier otro caso:
        * Usa la lógica equivalente a:
          ## CORRER XGBOOST POR ASIGNATURA SIN PRE REQUISITO
        * Llama a:
          df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(..., tiene_prereq=False)
          lista_prereq_usar = columnas_prereq_validas(...)

    Salida
    ------
    df_resultados_final : pd.DataFrame
        DataFrame con las observaciones por estudiante/asignatura, incluyendo:
        - Variables originales
        - Predicción de retiro (Clasificacion_XGB)
        - Predicción de nota (Prediccion_XGB)
        - Columnas de interpretaciones SHAP
    df_resultados_stats : pd.DataFrame
        Métricas de desempeño de los modelos de regresión por asignatura.
    df_resultados_class_retiro_stats : pd.DataFrame
        Métricas de desempeño de los modelos de clasificación de retiro por asignatura.
    """

    # ================================
    # 0) Inicialización de resultados
    # ================================
    col_usar = []
    var_objetivo = ""

    df_resultados_final = pd.DataFrame()
    df_resultados_stats = pd.DataFrame()
    df_resultados_class_retiro_stats = pd.DataFrame()

    # Lista de asignaturas a procesar
    asig_a_usar = df_usar["Cod materia curso"].unique().tolist()

    # Rutas para guardar modelos (ajusta si en tu notebook cambian)
    ruta_modelos_regresion = get_model_dir('xgboost', 'main', 'prediccion_nota')
    ruta_modelos_clasificacion = get_model_dir('xgboost', 'main', 'prediccion_retiro')

    for asig in asig_a_usar:
        try:
            print("\n" + "=" * 80)
            nombre_asig = df_usar[df_usar["Cod materia curso"] == asig]["Descripcion_Materia"].iloc[0]
            print(f"\n == Resultados para programa: {asig} - {nombre_asig} == \n")

            # Subconjunto por asignatura
            df_usar_filtrado = df_usar[df_usar["Cod materia curso"] == asig].copy()

            # ============================================================
            # 1) BLOQUE DIFERENCIADO: PREPROCESAMIENTO SEGÚN tipo_asig_prereq
            # ============================================================
            # Obtenemos el tipo de asignatura con/sin prerequisitos
            tipo_asig = None
            if "Observacion_Prerrequisito" in df_usar_filtrado.columns:
                tipo_series = df_usar_filtrado["Observacion_Prerrequisito"].dropna()
                if len(tipo_series) > 0:
                    tipo_asig = tipo_series.iloc[0]

            if tipo_asig == "Prerrequisito cumplido":
                # -------- LÓGICA: CON PRE REQUISITOS --------
                print("[Info] Asignatura con prerequisitos. Usando lógica CON PRE REQUISITOS.")

                # La variable objetivo se define dentro de renombrar_columnas
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado,
                    tiene_prereq=True
                )

                # Variable objetivo para clasificación (retiro)
                var_objetivo_clas = "Retiro_Asignatura_Cat"

                # Identificación de columnas de prerequisitos válidas (versión extendida)
                # Usa df_historial_asignaturas_nombres que ya tienes definido en el notebook
                lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext(
                    df_usar_filtrado,
                    df_historial_asignaturas_nombres,
                    0.8
                )

            else:
                # -------- LÓGICA: SIN PRE REQUISITOS --------
                print("[Info] Asignatura sin prerequisitos (o sin marca). Usando lógica SIN PRE REQUISITO.")

                # La variable objetivo se define dentro de renombrar_columnas
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado,
                    tiene_prereq=False
                )

                # Variable objetivo para clasificación (retiro)
                var_objetivo_clas = "Retiro_Asignatura_Cat"

                # Columnas de prerequisitos válidas (versión simple)
                lista_prereq_usar = columnas_prereq_validas(df_usar_filtrado, 0.8)

            # Agregar columnas de prerequisitos a la lista general de predictores
            col_usar = col_usar + lista_prereq_usar

            print(
                f"Columnas a usar ({len(col_usar)}): {col_usar} \n"
                f"  Número de filas a tener en cuenta: {len(df_usar_filtrado)}"
            )

            # ============================================================
            # 2) BLOQUE COMÚN: LIMPIEZA Y TIPOS
            # ============================================================

            # Columnas categóricas a convertir
            cols_to_category = [
                "programa",
                "sexo",
                "procedencia_categoria",
                "profesor_codigo",
                "Tipo_colegio",
                "Tipo_calendario",
            ]

            # Asegurar que la variable objetivo de clasificación sea int
            if var_objetivo_clas in df_usar_filtrado.columns:
                df_usar_filtrado[var_objetivo_clas] = df_usar_filtrado[var_objetivo_clas].astype(int)

            # Mantener solo las columnas solicitadas (y variables objetivo)
            cols_present = [c for c in col_usar if c in df_usar_filtrado.columns]

            if var_objetivo in df_usar_filtrado.columns and var_objetivo not in cols_present:
                cols_present.append(var_objetivo)

            if var_objetivo_clas in df_usar_filtrado.columns and var_objetivo_clas not in cols_present:
                cols_present.append(var_objetivo_clas)

            faltantes = [c for c in col_usar if c not in df_usar_filtrado.columns]
            if faltantes:
                print(f"[Aviso] Algunas columnas de col_usar no existen y se omiten: {faltantes}")

            # Reducir al subconjunto de columnas presentes
            df_usar_filtrado = df_usar_filtrado[cols_present].copy()
            print(
                f"[Info] df_usar_filtrado reducido a {df_usar_filtrado.shape[1]} columnas "
                f"y {len(df_usar_filtrado)} filas"
            )

            # Eliminar filas según tu lógica actual
            df_usar_filtrado, num_filas_eliminadas = eliminar_filas_por_columna(df_usar_filtrado)
            print(f"[Info] Filas eliminadas por condición: {num_filas_eliminadas}")

            # Convertir columnas categóricas
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            # Actualizar col_usar eliminando las columnas faltantes
            col_usar = [elemento for elemento in col_usar if elemento not in faltantes]

            # Estadísticos de retiros
            num_estud_retiros = len(df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 1])
            num_total_estud = len(df_usar_filtrado)
            porc_retiros = num_estud_retiros / num_total_estud if num_total_estud > 0 else 0

            # ============================================================
            # 3) MODELO DE CLASIFICACIÓN DE RETIROS
            # ============================================================
            print(f"[Aviso] El porcentaje de retiros es ({porc_retiros:.2%}).")

            if porc_retiros > 0.03:
                print("\n-------------------Modelo de Clasificación de Retiros--------------------\n")
                print(col_usar)

                # Entrenar modelo de clasificación
                (
                    modelo_clasif,
                    df_pred_clasif,
                    metricas_clasif,
                    X_train_clasif,
                    X_test_clasif,
                    y_train_clasif,
                    y_test_clasif,
                    df_resultados_stats_class_temp,
                ) = entrenar_xgboost_classif(
                    df=df_usar_filtrado,
                    columna_objetivo=var_objetivo_clas,
                    columnas_predictores=col_usar,
                )

                col_prediccion_clasif = "Clasificacion_XGB"

                # Añadimos identificación de la asignatura al dataframe de métricas de clasificación
                df_resultados_stats_class_temp["Descripcion_Materia"] = nombre_asig
                df_resultados_stats_class_temp["Cod materia curso"] = asig

                # Acumular métricas de clasificación
                if not isinstance(df_resultados_class_retiro_stats, pd.DataFrame):
                    try:
                        df_resultados_class_retiro_stats = pd.DataFrame(df_resultados_class_retiro_stats)
                    except Exception:
                        df_resultados_class_retiro_stats = pd.DataFrame()

                df_resultados_class_retiro_stats = pd.concat(
                    [df_resultados_class_retiro_stats, df_resultados_stats_class_temp],
                    ignore_index=True,
                    sort=False,
                )

                # Añadir la predicción de retiro al dataframe de trabajo
                df_to_add = df_pred_clasif[col_prediccion_clasif].reindex(df_usar_filtrado.index)
                df_usar_filtrado = df_usar_filtrado.join(df_to_add)

                # Guardar modelo de clasificación
                guardar_modelo_xg(modelo_clasif, ruta_modelos_clasificacion + "/", f"{asig}")

            else:
                # No se entrena modelo de clasificación por bajo % de retiros
                print(
                    f"[Aviso] El porcentaje de retiros es muy bajo ({porc_retiros:.2%}). "
                    "No se entrena modelo de clasificación."
                )

                col_prediccion_clasif = "Clasificacion_XGB"
                df_usar_filtrado[col_prediccion_clasif] = 0

                if not isinstance(df_resultados_class_retiro_stats, pd.DataFrame):
                    try:
                        df_resultados_class_retiro_stats = pd.DataFrame(df_resultados_class_retiro_stats)
                    except Exception:
                        df_resultados_class_retiro_stats = pd.DataFrame()

                error_row_class = pd.DataFrame(
                    [
                        {
                            "mean": "error",
                            "std": "error",
                            "metric": (
                                "Modelo Retiros no generado por bajo "
                                f"porcentaje de retiros ({porc_retiros:.2%})"
                            ),
                            "Descripcion_Materia": nombre_asig,
                            "Cod materia curso": asig,
                        }
                    ]
                )

                df_resultados_class_retiro_stats = pd.concat(
                    [df_resultados_class_retiro_stats, error_row_class],
                    ignore_index=True,
                    sort=False,
                )

            # ============================================================
            # 4) MODELO DE REGRESIÓN DE NOTA FINAL
            # ============================================================

            # Para la nota, se usa solo el subconjunto de estudiantes que NO retiraron la asignatura
            df_usar_pred_nota_f = df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 0]

            print("\n-------------------Modelo de Regresión de Nota Final--------------------\n")

            (
                modelo,
                df_pred,
                metricas,
                X_train,
                X_test,
                y_train,
                y_test,
                df_resultados_stats_temp,
            ) = entrenar_xgboost_regresion(
                df=df_usar_pred_nota_f,
                columna_objetivo=var_objetivo,
                columnas_predictores=col_usar,
                tuning=False,
            )

            col_prediccion = "Prediccion_XGB"

            # Interpretabilidad global básica (función de plots que ya tienes)
            interpretar_xgboost_shap(modelo, X_test, col_usar, top_n=1, id_check=0)

            # ============================================================
            # 5) INTERPRETACIONES SHAP EN EL DATAFRAME DE PREDICCIONES
            # ============================================================
            cols_to_category = cols_to_category + ["interpretacion_general", "interpretacion_registro"]
            df_pred = cambiar_a_category(df_pred, cols_to_category)

            # Escribir interpretaciones SHAP a nivel general y por registro
            df_pred, importancia_media = escribir_interpretaciones_shap(
                modelo,
                df_pred[col_usar],
                df_pred,
                top_n=10,
                col_general="interpretacion_general",
                col_registro="interpretacion_registro",
            )

            # Añadir información de asignatura al dataframe de métricas de regresión
            df_resultados_stats_temp["Descripcion_Materia"] = nombre_asig
            df_resultados_stats_temp["Cod materia curso"] = asig

            # Acumular métricas de regresión
            if not isinstance(df_resultados_stats, pd.DataFrame):
                try:
                    df_resultados_stats = pd.DataFrame(df_resultados_stats)
                except Exception:
                    df_resultados_stats = pd.DataFrame()

            df_resultados_stats = pd.concat(
                [df_resultados_stats, df_resultados_stats_temp],
                ignore_index=True,
                sort=False,
            )

            # ============================================================
            # 6) POST-PROCESADO DE PREDICCIONES Y ENSAMBLE FINAL
            # ============================================================

            # Redondear predicciones, truncar a rango [ -1, 5 ]
            if col_prediccion in df_pred.columns:
                df_pred[col_prediccion] = df_pred[col_prediccion].round(2)
                df_pred[col_prediccion] = df_pred[col_prediccion].apply(lambda x: -1 if x < 0 else x)
                df_pred[col_prediccion] = df_pred[col_prediccion].apply(lambda x: 5 if x > 5 else x)

            # Añadir predicción de nota e interpretaciones al dataframe de trabajo
            cols_to_add_pred = [col_prediccion, "interpretacion_general", "interpretacion_registro"]
            df_to_add = df_pred[cols_to_add_pred].reindex(df_usar_filtrado.index)
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

            # Acumular resultados finales
            if not isinstance(df_resultados_final, pd.DataFrame):
                try:
                    df_resultados_final = pd.DataFrame(df_resultados_final)
                except Exception:
                    df_resultados_final = pd.DataFrame()

            df_resultados_final = pd.concat(
                [df_resultados_final, df_usar_filtrado],
                ignore_index=True,
                sort=False,
            )

            # Guardar modelo de regresión
            guardar_modelo_xg(modelo, ruta_modelos_regresion + "/", f"{asig}")

        except Exception as e:
            # ============================================================
            # 7) MANEJO DE ERRORES POR ASIGNATURA
            # ============================================================
            print(f"[Error] Falló la asignatura {asig}. {type(e).__name__}: {e}")

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

    # Retornar los tres dataframes de resultados
    return df_resultados_final, df_resultados_stats, df_resultados_class_retiro_stats
