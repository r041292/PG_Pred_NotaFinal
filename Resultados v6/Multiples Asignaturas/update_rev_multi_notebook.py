import json
import pathlib
import re
import textwrap


NOTEBOOK_PATH = pathlib.Path(
    r"C:\Users\Rubiel\OneDrive - Universidad del Norte\Maestria\Proyecto de grado\Prereq\Resultados v6\Multiples Asignaturas\rev_multiples_asig_rev_ia.ipynb"
)


def replace_func(src: str, name: str, new_def: str) -> str:
    pattern = re.compile(rf"def {name}\(.*?(?=\ndef |\Z)", re.S)
    new_src, count = pattern.subn(new_def.strip() + "\n\n", src, count=1)
    if count != 1:
        raise RuntimeError(f"No se pudo reemplazar la funcion {name}")
    return new_src


def ensure_insert(src: str, anchor: str, block: str) -> str:
    if block in src:
        return src
    if anchor not in src:
        raise RuntimeError("No se encontro anchor de insercion")
    return src.replace(anchor, anchor + block, 1)


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))

    cell6 = "".join(nb["cells"][5]["source"])
    helper6 = textwrap.dedent(
        """

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
    )
    cell6 = ensure_insert(cell6, "    return X, cat_cols\n", helper6)
    nb["cells"][5]["source"] = cell6.splitlines(keepends=True)

    cell12 = "".join(nb["cells"][11]["source"])
    cell12 = cell12.replace(
        "import os\n\nimport json\n\nimport time\n\nVALID_MODELOS",
        "import os\n\nimport io\n\nimport json\n\nimport time\n\nfrom contextlib import redirect_stdout\n\nVALID_MODELOS",
    )
    helper12_anchor = (
        "def guardar_features_meta(paths, features=None, meta=None):\n"
        "    os.makedirs(os.path.dirname(paths['model']), exist_ok=True)\n"
        "    if features:\n"
        "        with open(paths['features'], 'w', encoding='utf-8') as f:\n"
        "            json.dump(list(features), f, ensure_ascii=False)\n"
        "    if meta:\n"
        "        with open(paths['meta'], 'w', encoding='utf-8') as f:\n"
        "            json.dump(meta, f, ensure_ascii=False)\n"
    )
    helper12 = textwrap.dedent(
        """

        def ejecutar_y_capturar_artefactos(func, *args, **kwargs):
            plt.close('all')
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                resultado = func(*args, **kwargs)
            texto = buffer.getvalue()
            if texto:
                print(texto, end='')
            figuras = [plt.figure(num) for num in plt.get_fignums()]
            return resultado, texto, figuras

        def _normalizar_nombre_artefacto(nombre):
            nombre = str(nombre)
            nombre = re.sub(r'[^A-Za-z0-9._-]+', '_', nombre)
            return nombre.strip('_') or 'artefacto'

        def guardar_artefactos_modelo(paths, cod_asig=None, texto=None, tablas=None, metricas=None, figuras=None):
            base_dir = os.path.dirname(paths['model'])
            dir_texto = os.path.join(base_dir, 'texto')
            dir_imagenes = os.path.join(base_dir, 'imagenes')
            os.makedirs(dir_texto, exist_ok=True)
            os.makedirs(dir_imagenes, exist_ok=True)

            prefijo = _normalizar_nombre_artefacto(cod_asig or pathlib.Path(paths['model']).stem)

            if texto:
                ruta_texto = os.path.join(dir_texto, f'{prefijo}_salida.txt')
                with open(ruta_texto, 'w', encoding='utf-8') as f:
                    f.write(texto)

            if metricas is not None:
                ruta_metricas = os.path.join(dir_texto, f'{prefijo}_metricas.json')
                with open(ruta_metricas, 'w', encoding='utf-8') as f:
                    json.dump(metricas, f, ensure_ascii=False, indent=2, default=str)

            tablas = tablas or {}
            for nombre, valor in tablas.items():
                nombre_norm = _normalizar_nombre_artefacto(nombre)
                ruta_base = os.path.join(dir_texto, f'{prefijo}_{nombre_norm}')
                if isinstance(valor, pd.DataFrame):
                    valor.to_csv(ruta_base + '.csv', index=True, encoding='utf-8-sig')
                elif isinstance(valor, pd.Series):
                    valor.to_frame(name='valor').to_csv(ruta_base + '.csv', index=True, encoding='utf-8-sig')
                elif isinstance(valor, dict):
                    with open(ruta_base + '.json', 'w', encoding='utf-8') as f:
                        json.dump(valor, f, ensure_ascii=False, indent=2, default=str)
                else:
                    with open(ruta_base + '.txt', 'w', encoding='utf-8') as f:
                        f.write(str(valor))

            figuras = figuras or []
            for idx, figura in enumerate(figuras, start=1):
                ruta_figura = os.path.join(dir_imagenes, f'{prefijo}_figura_{idx:02d}.png')
                figura.savefig(ruta_figura, bbox_inches='tight', dpi=200)
                plt.close(figura)
        """
    )
    cell12 = ensure_insert(cell12, helper12_anchor, helper12)

    cell12 = replace_func(
        cell12,
        "guardar_xgboost_modelo",
        textwrap.dedent(
            """
            def guardar_xgboost_modelo(modelo, ruta_modelos_regresion=None, ruta_modelos_clasificacion=None, cod_asig=None, variant='main', artefactos=None):
                task = 'prediccion_nota' if ruta_modelos_regresion is not None else 'prediccion_retiro'
                ensure_model_dir('xgboost', variant, task)
                paths = get_model_paths(cod_asig, 'xgboost', variant, task)
                modelo.save_model(paths['model'])
                features = _infer_features_from_model(modelo, 'xgboost')
                meta = {
                    'modelo': 'xgboost',
                    'variant': variant,
                    'task': task,
                    'asignatura': cod_asig,
                    'timestamp': time.time(),
                }
                guardar_features_meta(paths, features=features, meta=meta)
                if artefactos:
                    guardar_artefactos_modelo(paths, cod_asig=cod_asig, **artefactos)
                print("[Resultados Funcion : INFO] ", '[Info] Modelo XGBoost guardado en: ' + paths['model'])
                return paths['model']
            """
        ),
    )
    cell12 = replace_func(
        cell12,
        "guardar_sklearn_modelo",
        textwrap.dedent(
            """
            def guardar_sklearn_modelo(modelo, ruta_modelos_regresion=None, ruta_modelos_clasificacion=None, cod_asig=None, prefijo='rf', variant='main', artefactos=None):
                task = 'prediccion_nota' if ruta_modelos_regresion is not None else 'prediccion_retiro'
                ensure_model_dir('rf', variant, task)
                paths = get_model_paths(cod_asig, 'rf', variant, task)
                joblib.dump(modelo, paths['model'])
                features = _infer_features_from_model(modelo, 'rf')
                meta = {
                    'modelo': 'rf',
                    'variant': variant,
                    'task': task,
                    'asignatura': cod_asig,
                    'timestamp': time.time(),
                }
                guardar_features_meta(paths, features=features, meta=meta)
                if artefactos:
                    guardar_artefactos_modelo(paths, cod_asig=cod_asig, **artefactos)
                print("[Resultados Funcion : INFO] ", '[Info] Modelo RF guardado en: ' + paths['model'])
                return paths['model']
            """
        ),
    )
    cell12 = replace_func(
        cell12,
        "guardar_catboost_modelo",
        textwrap.dedent(
            """
            def guardar_catboost_modelo(modelo, ruta_modelos_regresion=None, ruta_modelos_clasificacion=None, cod_asig=None, variant='main', artefactos=None):
                task = 'prediccion_nota' if ruta_modelos_regresion is not None else 'prediccion_retiro'
                ensure_model_dir('catboost', variant, task)
                paths = get_model_paths(cod_asig, 'catboost', variant, task)
                modelo.save_model(paths['model'])
                features = _infer_features_from_model(modelo, 'catboost')
                meta = {
                    'modelo': 'catboost',
                    'variant': variant,
                    'task': task,
                    'asignatura': cod_asig,
                    'timestamp': time.time(),
                }
                guardar_features_meta(paths, features=features, meta=meta)
                if artefactos:
                    guardar_artefactos_modelo(paths, cod_asig=cod_asig, **artefactos)
                print("[Resultados Funcion : INFO] ", '[Info] Modelo CatBoost guardado en: ' + paths['model'])
                return paths['model']
            """
        ),
    )
    nb["cells"][11]["source"] = cell12.splitlines(keepends=True)

    cell8 = "".join(nb["cells"][7]["source"])
    cell8 = replace_func(
        cell8,
        "entrenar_xgboost_regresion",
        textwrap.dedent(
            """
            def entrenar_xgboost_regresion(df, columna_objetivo, columnas_predictores, tuning=True, cols_to_category=None, generar_interpretacion=True, top_n_interpretacion=10):
                X = df[columnas_predictores]
                y = df[columna_objetivo]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                modelo = XGBRegressor(
                    objective="reg:squarederror",
                    enable_categorical=True,
                    random_state=42,
                    subsample=0.8,
                    reg_lambda=3,
                    reg_alpha=0.01,
                    n_estimators=800,
                    min_child_weight=3,
                    max_depth=10,
                    learning_rate=0.05,
                    gamma=0,
                    colsample_bytree=0.8
                )

                if tuning:
                    print("[Resultados Funcion : INFO] ", "\\n[Regresion XGB] Iniciando optimizacion de hiperparametros con RandomSearchCV...\\n")
                    modelo, best_params, best_score = optimizar_xgboost_random(
                        modelo, X_train, y_train, n_iter=300, cv=5
                    )
                    print("[Resultados Funcion : INFO] ", "\\n[Regresion XGB] Optimizacion finalizada.")
                    print("[Resultados Funcion : INFO] ", "Mejores parametros aplicados al modelo:", best_params)
                    print("[Resultados Funcion : INFO] ", "Mejor score validacion cruzada (RMSE):", best_score)

                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                metricas = {"RMSE": rmse, "MAE": mae, "R2": r2}
                metricas_test_df = pd.DataFrame(metricas, index=["Valores"]).T

                print("[Resultados Funcion : INFO] ", "\\n[Regresion XGB] Metricas de evaluacion del modelo en test:\\n")
                print("[Resultados Funcion : INFO] ", metricas_test_df)

                resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv(modelo, X, y, n_splits=5, margen=0.3)

                df_resultados = df.copy()
                df_resultados["Prediccion_XGB"] = modelo.predict(X)

                importancia_media_xgb = None
                if generar_interpretacion:
                    cols_interpret_xgb = ["interpretacion_general_xgb", "interpretacion_registro_xgb"]
                    df_resultados = cambiar_a_category(df_resultados, list(cols_to_category or []) + cols_interpret_xgb)
                    print("[Resultados Funcion : INFO] ", "[Regresion XGB] Insertando interpretaciones SHAP en el dataframe de predicciones:")
                    df_resultados, importancia_media_xgb = escribir_interpretaciones_shap_xgb(
                        modelo,
                        df_resultados[columnas_predictores],
                        df_resultados,
                        top_n=top_n_interpretacion,
                        col_general=cols_interpret_xgb[0],
                        col_registro=cols_interpret_xgb[1],
                    )
                    interpretar_xgboost_shap(modelo, X_test, columnas_predictores, top_n=1, id_check=0)

                artefactos = {
                    'tablas': {
                        'metricas_test': metricas_test_df,
                        'metricas_cv': resultados_cv_df,
                    },
                    'metricas': metricas,
                }
                if importancia_media_xgb is not None:
                    artefactos['tablas']['importancia_shap'] = importancia_media_xgb

                return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df, artefactos
            """
        ),
    )
    cell8 = replace_func(
        cell8,
        "entrenar_xgboost_classif",
        textwrap.dedent(
            """
            def entrenar_xgboost_classif(df, columna_objetivo, columnas_predictores, tuning=False, cambiar_threshold=False):
                X = df[columnas_predictores]
                y = df[columna_objetivo]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                ratio_neg_to_pos = (y == 0).sum() / (y == 1).sum()
                print("[Resultados Funcion : INFO] ", f"El ratio de negativos / positivos es {ratio_neg_to_pos} ")

                modelo = XGBClassifier(
                    objective='binary:logistic',
                    enable_categorical=True,
                    eval_metric='logloss',
                    random_state=42,
                    scale_pos_weight=ratio_neg_to_pos,
                )

                if tuning:
                    print("[Resultados Funcion : INFO] ", "\\n[Clasificacion XGB] Iniciando optimizacion de hiperparametros con RandomSearchCV...\\n")
                    modelo, best_params, best_score = optimizar_xgboost_random(
                        modelo, X_train, y_train, n_iter=800, cv=7
                    )
                    print("[Resultados Funcion : INFO] ", "\\n[Clasificacion XGB] Optimizacion finalizada.")
                    print("[Resultados Funcion : INFO] ", "Mejores parametros aplicados al modelo:", best_params)
                    print("[Resultados Funcion : INFO] ", "Mejor score validacion cruzada:", best_score)

                modelo.fit(X_train, y_train)

                if cambiar_threshold:
                    th_opt, rec_opt, prec_opt = buscar_mejor_threshold(
                        modelo, X_train, y_train, min_precision=0.2
                    )
                    print("[Resultados Funcion : INFO] ", "\\n[Clasificacion XGB] Mejor threshold encontrado en:")
                    print("[Resultados Funcion : INFO] ", th_opt, rec_opt, prec_opt)
                    y_prob = modelo.predict_proba(X_test)[:, 1]
                    y_pred = (y_prob >= th_opt).astype(int)
                else:
                    y_pred = modelo.predict(X_test)
                    y_prob = modelo.predict_proba(X_test)[:, 1]

                print("[Resultados Funcion : INFO] ", "\\n[Clasificacion XGB] Metricas de evaluacion del modelo en test:\\n")
                class_report = classification_report(y_test, y_pred, output_dict=True)
                pprint(class_report)
                auc_score = roc_auc_score(y_test, y_prob)
                print("[Resultados Funcion : INFO] ", "AUC:", auc_score)
                c_matrix = confusion_matrix(y_test, y_pred)

                disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=["No Retiro", "Retiro"])
                disp.plot(cmap="Blues")
                plt.title("Matriz de Confusion")
                plt.show()

                metricas = {
                    'precision': class_report['macro avg']['precision'],
                    'recall': class_report['macro avg']['recall'],
                    'f1-score': class_report['macro avg']['f1-score'],
                    'auc': auc_score,
                }
                metricas_df = pd.DataFrame(metricas, index=["Metricas_Generales"]).T
                class_report_df = pd.DataFrame(class_report).T
                confusion_df = pd.DataFrame(c_matrix, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])
                print("[Resultados Funcion : INFO] ", metricas_df)

                resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv_clasificacion(modelo, X, y, n_splits=5, margen=0.3)

                df_resultados = df.copy()
                df_resultados["Clasificacion_XGB"] = modelo.predict(X)

                artefactos = {
                    'tablas': {
                        'metricas_test': metricas_df,
                        'classification_report': class_report_df,
                        'matriz_confusion': confusion_df,
                        'metricas_cv': resultados_cv_df,
                    },
                    'metricas': metricas,
                }

                return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df, artefactos
            """
        ),
    )
    nb["cells"][7]["source"] = cell8.splitlines(keepends=True)

    cell14 = "".join(nb["cells"][13]["source"])
    replacements = {
        "entrenar_rf_regresion": """
def entrenar_rf_regresion(df, columna_objetivo, columnas_predictores, tuning=True):
    X, _ = preparar_X_numerico(df, columnas_predictores)
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
    )

    if tuning:
        print("[Resultados Funcion : INFO] ", "\\n[RF] Iniciando optimizacion de hiperparametros con RandomSearchCV...\\n")
        modelo, best_params, best_score = optimizar_rf_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("[Resultados Funcion : INFO] ", "\\n[RF] Optimizacion finalizada.")
        print("[Resultados Funcion : INFO] ", "Mejores parametros aplicados al modelo:", best_params)
        print("[Resultados Funcion : INFO] ", "Mejor score validacion cruzada (RMSE):", best_score)

    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metricas = {"RMSE": rmse, "MAE": mae, "R2": r2}
    metricas_test_df = pd.DataFrame(metricas, index=["Valores"]).T

    print("[Resultados Funcion : INFO] ", "\\n[RF] Metricas de evaluacion del modelo en test:\\n")
    print("[Resultados Funcion : INFO] ", metricas_test_df)

    resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv(modelo, X, y, n_splits=5, margen=0.3)

    df_resultados = df.copy()
    df_resultados["Prediccion_RF"] = modelo.predict(X)

    artefactos = {
        'tablas': {
            'metricas_test': metricas_test_df,
            'metricas_cv': resultados_cv_df,
        },
        'metricas': metricas,
    }

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df, artefactos
""",
        "entrenar_rf_classif": """
def entrenar_rf_classif(df, columna_objetivo, columnas_predictores, tuning=False, cambiar_threshold=False):
    X, _ = preparar_X_numerico(df, columnas_predictores)
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ratio_neg_to_pos = (y == 0).sum() / (y == 1).sum()
    print("[Resultados Funcion : INFO] ", f"El ratio de negativos / positivos es {ratio_neg_to_pos} ")

    modelo = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    if tuning:
        print("[Resultados Funcion : INFO] ", "\\n[RF] Iniciando optimizacion de hiperparametros con RandomSearchCV...\\n")
        modelo, best_params, best_score = optimizar_rf_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("[Resultados Funcion : INFO] ", "\\n[RF] Optimizacion finalizada.")
        print("[Resultados Funcion : INFO] ", "Mejores parametros aplicados al modelo:", best_params)
        print("[Resultados Funcion : INFO] ", "Mejor score validacion cruzada:", best_score)

    modelo.fit(X_train, y_train)

    if cambiar_threshold:
        th_opt, rec_opt, prec_opt = buscar_mejor_threshold(
            modelo, X_train, y_train, min_precision=0.2
        )
        print("[Resultados Funcion : INFO] ", "\\n[RF] Mejor threshold encontrado en:")
        print("[Resultados Funcion : INFO] ", th_opt, rec_opt, prec_opt)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= th_opt).astype(int)
    else:
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

    print("[Resultados Funcion : INFO] ", "\\n[RF] Metricas de evaluacion del modelo en test:\\n")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    pprint(class_report)
    auc_score = roc_auc_score(y_test, y_prob)
    print("[Resultados Funcion : INFO] ", "AUC:", auc_score)
    c_matrix = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=["No Retiro", "Retiro"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusion")
    plt.show()

    metricas = {
        'precision': class_report['macro avg']['precision'],
        'recall': class_report['macro avg']['recall'],
        'f1-score': class_report['macro avg']['f1-score'],
        'auc': auc_score,
    }
    metricas_df = pd.DataFrame(metricas, index=["Metricas_Generales"]).T
    class_report_df = pd.DataFrame(class_report).T
    confusion_df = pd.DataFrame(c_matrix, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])
    print("[Resultados Funcion : INFO] ", metricas_df)

    resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv_clasificacion(
        modelo, X, y, n_splits=5, random_state=42
    )

    df_resultados = df.copy()
    df_resultados["Clasificacion_RF"] = modelo.predict(X)

    artefactos = {
        'tablas': {
            'metricas_test': metricas_df,
            'classification_report': class_report_df,
            'matriz_confusion': confusion_df,
            'metricas_cv': resultados_cv_df,
        },
        'metricas': metricas,
    }

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df, artefactos
""",
        "entrenar_catboost_regresion": """
def entrenar_catboost_regresion(df, columna_objetivo, columnas_predictores, tuning=True, cols_to_category=None, generar_interpretacion=True, top_n_interpretacion=10):
    X = df[columnas_predictores].copy()
    y = df[columna_objetivo]

    cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'object']
    cat_features = [X.columns.get_loc(c) for c in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )

    if tuning:
        print("[Resultados Funcion : INFO] ", "[CAT] Iniciando optimizacion de hiperparametros con RandomSearchCV...")
        modelo, best_params, best_score = optimizar_catboost_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("[Resultados Funcion : INFO] ", "[CAT] Optimizacion finalizada.")
        print("[Resultados Funcion : INFO] ", "Mejores parametros aplicados al modelo:", best_params)
        print("[Resultados Funcion : INFO] ", "Mejor score validacion cruzada (RMSE):", best_score)

    if cat_features:
        modelo.fit(X_train, y_train, cat_features=cat_features)
    else:
        modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metricas = {"RMSE": rmse, "MAE": mae, "R2": r2}
    metricas_test_df = pd.DataFrame(metricas, index=["Valores"]).T

    print("[Resultados Funcion : INFO] ", "[CAT] Metricas de evaluacion del modelo en test:")
    print("[Resultados Funcion : INFO] ", metricas_test_df)

    if cat_features:
        resultados_cv, resultados_cv_df = evaluar_catboost_cv_regresion(
            modelo, X, y, cat_features=cat_features, n_splits=5, random_state=42
        )
    else:
        resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv(modelo, X, y, n_splits=5, margen=0.3)

    df_resultados = df.copy()
    df_resultados["Prediccion_CAT"] = modelo.predict(X)

    importancia_media_cat = None
    if generar_interpretacion:
        cols_interpret_cat = ["interpretacion_general_cat", "interpretacion_registro_cat"]
        df_resultados = cambiar_a_category(df_resultados, list(cols_to_category or []) + cols_interpret_cat)
        print("[Resultados Funcion : INFO] ", "[CAT] Insertando interpretaciones SHAP en el dataframe de predicciones:")
        df_resultados, importancia_media_cat = escribir_interpretaciones_shap_catboost(
            modelo,
            df_resultados[columnas_predictores],
            df_resultados,
            top_n=top_n_interpretacion,
            col_general=cols_interpret_cat[0],
            col_registro=cols_interpret_cat[1],
        )
        interpretar_catboost_shap(modelo, X_test, columnas_predictores, top_n=1, id_check=0)

    artefactos = {
        'tablas': {
            'metricas_test': metricas_test_df,
            'metricas_cv': resultados_cv_df,
        },
        'metricas': metricas,
    }
    if importancia_media_cat is not None:
        artefactos['tablas']['importancia_shap'] = importancia_media_cat

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df, artefactos
""",
        "entrenar_catboost_classif": """
def entrenar_catboost_classif(df, columna_objetivo, columnas_predictores, tuning=False, cambiar_threshold=False):
    X = df[columnas_predictores].copy()
    y = df[columna_objetivo]

    cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'object']
    cat_features = [X.columns.get_loc(c) for c in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ratio_neg_to_pos = (y == 0).sum() / (y == 1).sum()
    print("[Resultados Funcion : INFO] ", f"El ratio de negativos / positivos es {ratio_neg_to_pos} ")

    modelo = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        scale_pos_weight=ratio_neg_to_pos,
    )

    if tuning:
        print("[Resultados Funcion : INFO] ", "[CAT] Iniciando optimizacion de hiperparametros con RandomSearchCV...")
        modelo, best_params, best_score = optimizar_catboost_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("[Resultados Funcion : INFO] ", "[CAT] Optimizacion finalizada.")
        print("[Resultados Funcion : INFO] ", "Mejores parametros aplicados al modelo:", best_params)
        print("[Resultados Funcion : INFO] ", "Mejor score validacion cruzada:", best_score)

    if cat_features:
        modelo.fit(X_train, y_train, cat_features=cat_features)
    else:
        modelo.fit(X_train, y_train)

    if cambiar_threshold:
        th_opt, rec_opt, prec_opt = buscar_mejor_threshold(
            modelo, X_train, y_train, min_precision=0.2
        )
        print("[Resultados Funcion : INFO] ", "[CAT] Mejor threshold encontrado en:")
        print("[Resultados Funcion : INFO] ", th_opt, rec_opt, prec_opt)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= th_opt).astype(int)
    else:
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

    print("[Resultados Funcion : INFO] ", "[CAT] Metricas de evaluacion del modelo en test:")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    pprint(class_report)
    auc_score = roc_auc_score(y_test, y_prob)
    print("[Resultados Funcion : INFO] ", "AUC:", auc_score)
    c_matrix = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=["No Retiro", "Retiro"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusion")
    plt.show()

    metricas = {
        'precision': class_report['macro avg']['precision'],
        'recall': class_report['macro avg']['recall'],
        'f1-score': class_report['macro avg']['f1-score'],
        'auc': auc_score,
    }
    metricas_df = pd.DataFrame(metricas, index=["Metricas_Generales"]).T
    class_report_df = pd.DataFrame(class_report).T
    confusion_df = pd.DataFrame(c_matrix, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])
    print("[Resultados Funcion : INFO] ", metricas_df)

    if cat_features:
        print("[Resultados Funcion : INFO] ", "[CAT] CV omitido: CatBoost con cat_features no es compatible con cross_val_score sin wrapper.")
        resultados_cv = {}
        y_oof = None
        resultados_cv_df = pd.DataFrame([{
            'mean': 'error',
            'std': 'error',
            'metric': 'CV no calculado - cat_features'
        }])
    else:
        resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv_clasificacion(
            modelo, X, y, n_splits=5, random_state=42
        )

    df_resultados = df.copy()
    df_resultados["Clasificacion_CAT"] = modelo.predict(X)

    artefactos = {
        'tablas': {
            'metricas_test': metricas_df,
            'classification_report': class_report_df,
            'matriz_confusion': confusion_df,
            'metricas_cv': resultados_cv_df,
        },
        'metricas': metricas,
    }

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df, artefactos
""",
    }
    for name, code in replacements.items():
        cell14 = replace_func(cell14, name, textwrap.dedent(code))
    nb["cells"][13]["source"] = cell14.splitlines(keepends=True)

    cell16 = "".join(nb["cells"][15]["source"])
    cell16 = replace_func(
        cell16,
        "correr_modelos_multi_por_asignatura",
        textwrap.dedent(
            """
            def correr_modelos_multi_por_asignatura(df_usar, variant_modelo_generar='main'):
                col_usar = []
                var_objetivo = ''

                df_resultados_final = pd.DataFrame()
                df_resultados_stats = pd.DataFrame()
                df_resultados_class_retiro_stats = pd.DataFrame()

                asig_a_usar = df_usar['Cod materia curso'].unique().tolist()

                ruta_modelos_reg_xgb = get_model_dir('xgboost', variant_modelo_generar, 'prediccion_nota')
                ruta_modelos_cls_xgb = get_model_dir('xgboost', variant_modelo_generar, 'prediccion_retiro')
                ruta_modelos_reg_rf = get_model_dir('rf', variant_modelo_generar, 'prediccion_nota')
                ruta_modelos_cls_rf = get_model_dir('rf', variant_modelo_generar, 'prediccion_retiro')
                ruta_modelos_reg_cat = get_model_dir('catboost', variant_modelo_generar, 'prediccion_nota')
                ruta_modelos_cls_cat = get_model_dir('catboost', variant_modelo_generar, 'prediccion_retiro')

                print("[Resultados Funcion : INFO] ", ruta_modelos_reg_xgb, ruta_modelos_cls_xgb, ruta_modelos_reg_rf, ruta_modelos_cls_rf, ruta_modelos_reg_cat, ruta_modelos_cls_cat)

                for p in [ruta_modelos_reg_xgb, ruta_modelos_cls_xgb, ruta_modelos_reg_rf, ruta_modelos_cls_rf, ruta_modelos_reg_cat, ruta_modelos_cls_cat]:
                    os.makedirs(p, exist_ok=True)

                for asig in asig_a_usar:
                    try:
                        print("[Resultados Funcion : INFO] ", '\\n' + '=' * 80)
                        df_asig_original = df_usar[df_usar['Cod materia curso'] == asig].copy()
                        nombre_asig = df_asig_original['Descripcion_Materia'].iloc[0]
                        print("[Resultados Funcion : INFO] ", f'\\n == Resultados para programa: {asig} - {nombre_asig} == \\n')

                        df_usar_filtrado = df_asig_original.copy()
                        df_resultado_asig = df_asig_original.copy()

                        tipo_asig = None
                        if 'Observacion_Prerrequisito' in df_usar_filtrado.columns:
                            tipo_series = df_usar_filtrado['Observacion_Prerrequisito'].dropna()
                            if len(tipo_series) > 0:
                                tipo_asig = tipo_series.iloc[0]

                        if tipo_asig == 'Prerrequisito cumplido':
                            print("[Resultados Funcion : INFO] ", '[Info] Asignatura con prerequisitos. Usando logica CON PRE REQUISITOS.')
                            df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(df_usar_filtrado, tiene_prereq=True)
                            lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext(
                                df_usar_filtrado, df_historial_asignaturas_nombres, 0.8
                            )
                        else:
                            print("[Resultados Funcion : INFO] ", '[Info] Asignatura sin prerequisitos. Usando logica SIN PRE REQUISITOS.')
                            df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(df_usar_filtrado, tiene_prereq=False)
                            lista_prereq_usar = columnas_prereq_validas(df_usar_filtrado, 0.8)

                        col_usar = col_usar + lista_prereq_usar
                        print("[Resultados Funcion : INFO] ", f'Columnas a usar ({len(col_usar)}): {col_usar} \\n  Numero de filas a tener en cuenta: {len(df_usar_filtrado)}')

                        cols_to_category = ['programa', 'sexo', 'procedencia_categoria', 'profesor_codigo', 'Tipo_colegio', 'Tipo_calendario']
                        var_objetivo_clas = 'Retiro_Asignatura_Cat'

                        df_usar_filtrado, col_usar, faltantes = preparar_dataframe_para_modelado(
                            df_usar_filtrado, col_usar, var_objetivo, var_objetivo_clas
                        )

                        df_usar_filtrado, num_filas_eliminadas = eliminar_filas_por_columna(df_usar_filtrado)
                        print("[Resultados Funcion : INFO] ", f'[Info] Filas eliminadas por condicion: {num_filas_eliminadas}')
                        df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

                        if variant_modelo_generar == 'fallback':
                            df_usar_filtrado = eliminar_columnas_prof(df_usar_filtrado)
                            col_usar = eliminar_columnas_prof_list(col_usar)
                            print("[Resultados Funcion : INFO] ", f'[Info] Tipo de modelo: Fallback. Columnas eliminadas por ser del profesor. Nuevas columnas a usar: {col_usar}')

                        num_estud_retiros = len(df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 1])
                        num_total_estud = len(df_usar_filtrado)
                        porc_retiros = num_estud_retiros / num_total_estud if num_total_estud else 0
                        print("[Resultados Funcion : INFO] ", f'[Aviso] El porcentaje de retiros es ({porc_retiros:.2%}).')

                        if porc_retiros > 0.03:
                            print("[Resultados Funcion : INFO] ", '\\n-------------------Modelo de Clasificacion de Retiros (XGB)--------------------\\n')
                            resultado_xgb_cls, texto_xgb_cls, figuras_xgb_cls = ejecutar_y_capturar_artefactos(
                                entrenar_xgboost_classif,
                                df=df_usar_filtrado,
                                columna_objetivo=var_objetivo_clas,
                                columnas_predictores=col_usar,
                            )
                            modelo_clasif_xgb, df_pred_clasif_xgb, metricas_clasif_xgb, X_train_clasif_xgb, X_test_clasif_xgb, y_train_clasif_xgb, y_test_clasif_xgb, df_resultados_stats_class_temp_xgb, artefactos_xgb_cls = resultado_xgb_cls
                            df_resultado_asig = anexar_columnas_por_indice(df_resultado_asig, df_pred_clasif_xgb, ['Clasificacion_XGB'])
                            df_resultados_stats_class_temp_xgb['Descripcion_Materia'] = nombre_asig
                            df_resultados_stats_class_temp_xgb['Cod materia curso'] = asig
                            df_resultados_stats_class_temp_xgb['Modelo'] = 'XGB'
                            df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp_xgb], axis=0)
                            artefactos_xgb_cls['texto'] = texto_xgb_cls
                            artefactos_xgb_cls['figuras'] = figuras_xgb_cls
                            guardar_xgboost_modelo(modelo_clasif_xgb, ruta_modelos_clasificacion=ruta_modelos_cls_xgb, cod_asig=asig, variant=variant_modelo_generar, artefactos=artefactos_xgb_cls)

                            print("[Resultados Funcion : INFO] ", '\\n-------------------Modelo de Clasificacion de Retiros (RF)--------------------\\n')
                            resultado_rf_cls, texto_rf_cls, figuras_rf_cls = ejecutar_y_capturar_artefactos(
                                entrenar_rf_classif,
                                df=df_usar_filtrado,
                                columna_objetivo=var_objetivo_clas,
                                columnas_predictores=col_usar,
                            )
                            modelo_clasif_rf, df_pred_clasif_rf, metricas_clasif_rf, X_train_clasif_rf, X_test_clasif_rf, y_train_clasif_rf, y_test_clasif_rf, df_resultados_stats_class_temp_rf, artefactos_rf_cls = resultado_rf_cls
                            df_resultado_asig = anexar_columnas_por_indice(df_resultado_asig, df_pred_clasif_rf, ['Clasificacion_RF'])
                            df_resultados_stats_class_temp_rf['Descripcion_Materia'] = nombre_asig
                            df_resultados_stats_class_temp_rf['Cod materia curso'] = asig
                            df_resultados_stats_class_temp_rf['Modelo'] = 'RF'
                            df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp_rf], axis=0)
                            artefactos_rf_cls['texto'] = texto_rf_cls
                            artefactos_rf_cls['figuras'] = figuras_rf_cls
                            guardar_sklearn_modelo(modelo_clasif_rf, ruta_modelos_clasificacion=ruta_modelos_cls_rf, cod_asig=asig, prefijo='rf', variant=variant_modelo_generar, artefactos=artefactos_rf_cls)

                            print("[Resultados Funcion : INFO] ", '\\n-------------------Modelo de Clasificacion de Retiros (CAT)--------------------\\n')
                            resultado_cat_cls, texto_cat_cls, figuras_cat_cls = ejecutar_y_capturar_artefactos(
                                entrenar_catboost_classif,
                                df=df_usar_filtrado,
                                columna_objetivo=var_objetivo_clas,
                                columnas_predictores=col_usar,
                            )
                            modelo_clasif_cat, df_pred_clasif_cat, metricas_clasif_cat, X_train_clasif_cat, X_test_clasif_cat, y_train_clasif_cat, y_test_clasif_cat, df_resultados_stats_class_temp_cat, artefactos_cat_cls = resultado_cat_cls
                            df_resultado_asig = anexar_columnas_por_indice(df_resultado_asig, df_pred_clasif_cat, ['Clasificacion_CAT'])
                            df_resultados_stats_class_temp_cat['Descripcion_Materia'] = nombre_asig
                            df_resultados_stats_class_temp_cat['Cod materia curso'] = asig
                            df_resultados_stats_class_temp_cat['Modelo'] = 'CAT'
                            df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp_cat], axis=0)
                            artefactos_cat_cls['texto'] = texto_cat_cls
                            artefactos_cat_cls['figuras'] = figuras_cat_cls
                            guardar_catboost_modelo(modelo_clasif_cat, ruta_modelos_clasificacion=ruta_modelos_cls_cat, cod_asig=asig, variant=variant_modelo_generar, artefactos=artefactos_cat_cls)
                        else:
                            print("[Resultados Funcion : INFO] ", f'[Aviso] El porcentaje de retiros es muy bajo ({porc_retiros:.2%}). No se entrena modelo de clasificacion.')
                            df_resultado_asig['Clasificacion_XGB'] = 0
                            df_resultado_asig['Clasificacion_RF'] = 0
                            df_resultado_asig['Clasificacion_CAT'] = 0
                            for modelo_nombre in ['XGB', 'RF', 'CAT']:
                                error_row_class = pd.DataFrame([{
                                    'mean': 'error',
                                    'std': 'error',
                                    'metric': f'Modelo Retiros no generado por bajo porcentaje de retiros ({porc_retiros:.2%})',
                                    'Descripcion_Materia': nombre_asig,
                                    'Cod materia curso': asig,
                                    'Modelo': modelo_nombre,
                                }])
                                df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, error_row_class], axis=0)

                        df_usar_pred_nota_f = df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 0]

                        print("[Resultados Funcion : INFO] ", '\\n-------------------Modelo de Regresion de Nota (XGB)--------------------\\n')
                        resultado_xgb_reg, texto_xgb_reg, figuras_xgb_reg = ejecutar_y_capturar_artefactos(
                            entrenar_xgboost_regresion,
                            df=df_usar_pred_nota_f,
                            columna_objetivo=var_objetivo,
                            columnas_predictores=col_usar,
                            tuning=False,
                            cols_to_category=cols_to_category,
                            generar_interpretacion=True,
                        )
                        modelo_xgb, df_pred_xgb, metricas_xgb, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, df_resultados_stats_temp_xgb, artefactos_xgb_reg = resultado_xgb_reg
                        df_resultado_asig = anexar_columnas_por_indice(df_resultado_asig, df_pred_xgb, ['Prediccion_XGB', 'interpretacion_general_xgb', 'interpretacion_registro_xgb'])
                        df_resultados_stats_temp_xgb['Descripcion_Materia'] = nombre_asig
                        df_resultados_stats_temp_xgb['Cod materia curso'] = asig
                        df_resultados_stats_temp_xgb['Modelo'] = 'XGB'
                        df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp_xgb], axis=0)
                        artefactos_xgb_reg['texto'] = texto_xgb_reg
                        artefactos_xgb_reg['figuras'] = figuras_xgb_reg
                        guardar_xgboost_modelo(modelo_xgb, ruta_modelos_regresion=ruta_modelos_reg_xgb, cod_asig=asig, variant=variant_modelo_generar, artefactos=artefactos_xgb_reg)

                        print("[Resultados Funcion : INFO] ", '\\n-------------------Modelo de Regresion de Nota (RF)--------------------\\n')
                        resultado_rf_reg, texto_rf_reg, figuras_rf_reg = ejecutar_y_capturar_artefactos(
                            entrenar_rf_regresion,
                            df=df_usar_pred_nota_f,
                            columna_objetivo=var_objetivo,
                            columnas_predictores=col_usar,
                            tuning=False,
                        )
                        modelo_rf, df_pred_rf, metricas_rf, X_train_rf, X_test_rf, y_train_rf, y_test_rf, df_resultados_stats_temp_rf, artefactos_rf_reg = resultado_rf_reg
                        df_resultado_asig = anexar_columnas_por_indice(df_resultado_asig, df_pred_rf, ['Prediccion_RF'])
                        df_resultados_stats_temp_rf['Descripcion_Materia'] = nombre_asig
                        df_resultados_stats_temp_rf['Cod materia curso'] = asig
                        df_resultados_stats_temp_rf['Modelo'] = 'RF'
                        df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp_rf], axis=0)
                        artefactos_rf_reg['texto'] = texto_rf_reg
                        artefactos_rf_reg['figuras'] = figuras_rf_reg
                        guardar_sklearn_modelo(modelo_rf, ruta_modelos_regresion=ruta_modelos_reg_rf, cod_asig=asig, prefijo='rf', variant=variant_modelo_generar, artefactos=artefactos_rf_reg)

                        print("[Resultados Funcion : INFO] ", '\\n-------------------Modelo de Regresion de Nota (CAT)--------------------\\n')
                        resultado_cat_reg, texto_cat_reg, figuras_cat_reg = ejecutar_y_capturar_artefactos(
                            entrenar_catboost_regresion,
                            df=df_usar_pred_nota_f,
                            columna_objetivo=var_objetivo,
                            columnas_predictores=col_usar,
                            tuning=False,
                            cols_to_category=cols_to_category,
                            generar_interpretacion=True,
                        )
                        modelo_cat, df_pred_cat, metricas_cat, X_train_cat, X_test_cat, y_train_cat, y_test_cat, df_resultados_stats_temp_cat, artefactos_cat_reg = resultado_cat_reg
                        df_resultado_asig = anexar_columnas_por_indice(df_resultado_asig, df_pred_cat, ['Prediccion_CAT', 'interpretacion_general_cat', 'interpretacion_registro_cat'])
                        df_resultados_stats_temp_cat['Descripcion_Materia'] = nombre_asig
                        df_resultados_stats_temp_cat['Cod materia curso'] = asig
                        df_resultados_stats_temp_cat['Modelo'] = 'CAT'
                        df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp_cat], axis=0)
                        artefactos_cat_reg['texto'] = texto_cat_reg
                        artefactos_cat_reg['figuras'] = figuras_cat_reg
                        guardar_catboost_modelo(modelo_cat, ruta_modelos_regresion=ruta_modelos_reg_cat, cod_asig=asig, variant=variant_modelo_generar, artefactos=artefactos_cat_reg)

                        df_resultado_asig = limitar_predicciones_nota(df_resultado_asig, ['Prediccion_XGB', 'Prediccion_RF', 'Prediccion_CAT'])
                        df_resultado_asig = cambiar_a_category(
                            df_resultado_asig,
                            cols_to_category + ['interpretacion_general_xgb', 'interpretacion_registro_xgb', 'interpretacion_general_cat', 'interpretacion_registro_cat']
                        )

                        if df_resultados_final is None or df_resultados_final.empty:
                            df_resultados_final = df_resultado_asig
                        else:
                            df_resultados_final = pd.concat([df_resultados_final, df_resultado_asig], axis=0, sort=False)

                    except Exception as e:
                        print("[Resultados Funcion : INFO] ", f'[Error] Fallo procesando asignatura {asig}: {e}')

                        try:
                            nombre_asig = df_usar[df_usar["Cod materia curso"] == asig]["Descripcion_Materia"].iloc[0]
                        except Exception:
                            nombre_asig = str(asig)

                        error_row = pd.DataFrame([{
                            'mean': 'error',
                            'std': 'error',
                            'metric': f'{type(e).__name__}: {e}',
                            'Descripcion_Materia': nombre_asig,
                            'Cod materia curso': asig,
                        }])

                        if not isinstance(df_resultados_stats, pd.DataFrame):
                            try:
                                df_resultados_stats = pd.DataFrame(df_resultados_stats)
                            except Exception:
                                df_resultados_stats = pd.DataFrame()

                        df_resultados_stats = pd.concat([df_resultados_stats, error_row], ignore_index=True, sort=False)
                        continue

                df_resultados_final = _reordenar_prereq_al_final(df_resultados_final)
                return df_resultados_final, df_resultados_stats, df_resultados_class_retiro_stats
            """
        ),
    )
    nb["cells"][15]["source"] = cell16.splitlines(keepends=True)

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("NOTEBOOK_UPDATED")


if __name__ == "__main__":
    main()
