# %% rf_catboost_helper
# Helper para preparar X numerico para modelos que no manejan categorias

def preparar_X_numerico(df, columnas_predictores):
    """
    Convierte columnas category/object a codigos numericos para modelos que no
    soportan categoricos nativamente (por ejemplo, RandomForest).
    """
    X = df[columnas_predictores].copy()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
    return X, cat_cols

# %% rf_catboost_modelos
# Funciones alternativas basadas en RandomForest y CatBoost
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from pprint import pprint
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    make_scorer,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from catboost import CatBoostRegressor, CatBoostClassifier


def optimizar_rf_random(modelo, X_train, y_train, n_iter=50, cv=5, random_state=42):
    param_dist = {
        "n_estimators": [200, 400, 800],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
    }

    recall_scorer = make_scorer(recall_margen, greater_is_better=True)

    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=recall_scorer,
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = -random_search.best_score_

    print("Mejores parametros encontrados:", best_params)
    print("Mejor score (RMSE):", best_score)

    return best_model, best_params, best_score


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
        print("\n[RF] Iniciando optimizacion de hiperparametros con RandomSearchCV...\n")
        modelo, best_params, best_score = optimizar_rf_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("\n[RF] Optimizacion finalizada.")
        print("Mejores parametros aplicados al modelo:", best_params)
        print("Mejor score validacion cruzada (RMSE):", best_score)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metricas = {"RMSE": rmse, "MAE": mae, "R2": r2}

    print("\n[RF] Metricas de evaluacion del modelo en test:\n")
    print(pd.DataFrame(metricas, index=["Valores"]).T)

    resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv(modelo, X, y, n_splits=5, margen=0.3)

    df_resultados = df.copy()
    df_resultados["Prediccion_RF"] = modelo.predict(X)

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df


def entrenar_rf_classif(df, columna_objetivo, columnas_predictores, tuning=False, cambiar_threshold=False):
    X, _ = preparar_X_numerico(df, columnas_predictores)
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ratio_neg_to_pos = (y == 0).sum() / (y == 1).sum()
    print(f"El ratio de negativos / positivos es {ratio_neg_to_pos} ")

    modelo = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    if tuning:
        print("\n[RF] Iniciando optimizacion de hiperparametros con RandomSearchCV...\n")
        modelo, best_params, best_score = optimizar_rf_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("\n[RF] Optimizacion finalizada.")
        print("Mejores parametros aplicados al modelo:", best_params)
        print("Mejor score validacion cruzada:", best_score)

    modelo.fit(X_train, y_train)

    if cambiar_threshold:
        th_opt, rec_opt, prec_opt = buscar_mejor_threshold(
            modelo, X_train, y_train, min_precision=0.2
        )
        print("\n[RF] Mejor threshold encontrado en:")
        print(th_opt, rec_opt, prec_opt)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= th_opt).astype(int)
    else:
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

    print("\n[RF] Metricas de evaluacion del modelo en test:\n")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    pprint(class_report)
    print("AUC:", roc_auc_score(y_test, y_prob))
    c_matrix = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=["No Retiro", "Retiro"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusion")
    plt.show()

    metricas = {
        "precision": class_report["macro avg"]["precision"],
        "recall": class_report["macro avg"]["recall"],
        "f1-score": class_report["macro avg"]["f1-score"],
    }

    print(pd.DataFrame(metricas, index=["Metricas_Generales"]).T)

    resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv_clasificacion(
        modelo, X, y, n_splits=5, random_state=42
    )

    df_resultados = df.copy()
    df_resultados["Clasificacion_RF"] = modelo.predict(X)

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df


def optimizar_catboost_random(modelo, X_train, y_train, n_iter=50, cv=5, random_state=42):
    param_dist = {
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.03, 0.1, 0.2],
        "n_estimators": [400, 800, 1200],
        "l2_leaf_reg": [1, 3, 5, 7],
    }

    recall_scorer = make_scorer(recall_margen, greater_is_better=True)

    random_search = RandomizedSearchCV(
        estimator=modelo,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=recall_scorer,
        cv=cv,
        verbose=2,
        random_state=random_state,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = -random_search.best_score_

    print("Mejores parametros encontrados:", best_params)
    print("Mejor score (RMSE):", best_score)

    return best_model, best_params, best_score


def entrenar_catboost_regresion(df, columna_objetivo, columnas_predictores, tuning=True):
    X, _ = preparar_X_numerico(df, columnas_predictores)
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = CatBoostRegressor(
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
    )

    if tuning:
        print("\n[CAT] Iniciando optimizacion de hiperparametros con RandomSearchCV...\n")
        modelo, best_params, best_score = optimizar_catboost_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("\n[CAT] Optimizacion finalizada.")
        print("Mejores parametros aplicados al modelo:", best_params)
        print("Mejor score validacion cruzada (RMSE):", best_score)

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metricas = {"RMSE": rmse, "MAE": mae, "R2": r2}

    print("\n[CAT] Metricas de evaluacion del modelo en test:\n")
    print(pd.DataFrame(metricas, index=["Valores"]).T)

    resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv(modelo, X, y, n_splits=5, margen=0.3)

    df_resultados = df.copy()
    df_resultados["Prediccion_CAT"] = modelo.predict(X)

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df


def entrenar_catboost_classif(df, columna_objetivo, columnas_predictores, tuning=False, cambiar_threshold=False):
    X, _ = preparar_X_numerico(df, columnas_predictores)
    y = df[columna_objetivo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    ratio_neg_to_pos = (y == 0).sum() / (y == 1).sum()
    print(f"El ratio de negativos / positivos es {ratio_neg_to_pos} ")

    modelo = CatBoostClassifier(
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
        scale_pos_weight=ratio_neg_to_pos,
    )

    if tuning:
        print("\n[CAT] Iniciando optimizacion de hiperparametros con RandomSearchCV...\n")
        modelo, best_params, best_score = optimizar_catboost_random(
            modelo, X_train, y_train, n_iter=200, cv=5
        )
        print("\n[CAT] Optimizacion finalizada.")
        print("Mejores parametros aplicados al modelo:", best_params)
        print("Mejor score validacion cruzada:", best_score)

    modelo.fit(X_train, y_train)

    if cambiar_threshold:
        th_opt, rec_opt, prec_opt = buscar_mejor_threshold(
            modelo, X_train, y_train, min_precision=0.2
        )
        print("\n[CAT] Mejor threshold encontrado en:")
        print(th_opt, rec_opt, prec_opt)
        y_prob = modelo.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= th_opt).astype(int)
    else:
        y_pred = modelo.predict(X_test)
        y_prob = modelo.predict_proba(X_test)[:, 1]

    print("\n[CAT] Metricas de evaluacion del modelo en test:\n")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    pprint(class_report)
    print("AUC:", roc_auc_score(y_test, y_prob))
    c_matrix = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=c_matrix, display_labels=["No Retiro", "Retiro"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusion")
    plt.show()

    metricas = {
        "precision": class_report["macro avg"]["precision"],
        "recall": class_report["macro avg"]["recall"],
        "f1-score": class_report["macro avg"]["f1-score"],
    }

    print(pd.DataFrame(metricas, index=["Metricas_Generales"]).T)

    resultados_cv, y_oof, resultados_cv_df = evaluar_con_cv_clasificacion(
        modelo, X, y, n_splits=5, random_state=42
    )

    df_resultados = df.copy()
    df_resultados["Clasificacion_CAT"] = modelo.predict(X)

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df


def interpretar_rf_shap(modelo, X, columnas_predictores, top_n=5, id_check=0):
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X)

    print("[RF] Generando summary plot global...")
    shap.summary_plot(shap_values, X, feature_names=columnas_predictores)

    importancia_media = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = importancia_media.head(top_n).index
    print(f"\n[RF] Generando dependence plots para las {top_n} variables mas importantes...\n")
    for feature in top_features:
        print(f"[RF] Dependence plot para variable: {feature}")
        shap.dependence_plot(feature, shap_values.values, X, feature_names=columnas_predictores)

    i = id_check
    print(f"[RF] Explicacion de la prediccion para el estudiante {i}")
    shap.plots.waterfall(shap_values[i])


def interpretar_catboost_shap(modelo, X, columnas_predictores, top_n=5, id_check=0):
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X)

    print("[CAT] Generando summary plot global...")
    shap.summary_plot(shap_values, X, feature_names=columnas_predictores)

    importancia_media = pd.Series(modelo.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    top_features = importancia_media.head(top_n).index
    print(f"\n[CAT] Generando dependence plots para las {top_n} variables mas importantes...\n")
    for feature in top_features:
        print(f"[CAT] Dependence plot para variable: {feature}")
        shap.dependence_plot(feature, shap_values.values, X, feature_names=columnas_predictores)

    i = id_check
    print(f"[CAT] Explicacion de la prediccion para el estudiante {i}")
    shap.plots.waterfall(shap_values[i])

# %% rf_catboost_pipelines
# Pipelines alternativos por asignatura (RF y CatBoost)

def correr_modelos_rf_por_asignatura(df_usar):
    col_usar = []
    var_objetivo = ""

    df_resultados_final = pd.DataFrame()
    df_resultados_stats = pd.DataFrame()
    df_resultados_class_retiro_stats = pd.DataFrame()

    asig_a_usar = df_usar["Cod materia curso"].unique().tolist()

    for asig in asig_a_usar:
        try:
            print("\n" + "=" * 80)
            nombre_asig = df_usar[df_usar["Cod materia curso"] == asig]["Descripcion_Materia"].iloc[0]
            print(f"\n == Resultados para programa: {asig} - {nombre_asig} == \n")

            df_usar_filtrado = df_usar[df_usar["Cod materia curso"] == asig].copy()

            tipo_asig = None
            if "Observacion_Prerrequisito" in df_usar_filtrado.columns:
                tipo_series = df_usar_filtrado["Observacion_Prerrequisito"].dropna()
                if len(tipo_series) > 0:
                    tipo_asig = tipo_series.iloc[0]

            if tipo_asig == "Prerrequisito cumplido":
                print("[Info] Asignatura con prerequisitos. Usando logica CON PRE REQUISITOS.")
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado, tiene_prereq=True
                )
                lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext_dif_prereq(
                    df_usar_filtrado, df_historial_asignaturas_nombres, 0.8
                )
            else:
                print("[Info] Asignatura sin prerequisitos. Usando logica SIN PRE REQUISITOS.")
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado, tiene_prereq=False
                )
                lista_prereq_usar = columnas_prereq_validas(df_usar_filtrado, 0.8)

            col_usar = col_usar + lista_prereq_usar

            print(f"Columnas a usar ({len(col_usar)}): {col_usar} \n  Numero de filas a tener en cuenta: {len(df_usar_filtrado)}")

            cols_to_category = [
                "programa",
                "sexo",
                "procedencia_categoria",
                "profesor_codigo",
                "Tipo_colegio",
                "Tipo_calendario",
            ]

            var_objetivo_clas = "Retiro_Asignatura_Cat"
            if var_objetivo_clas in df_usar_filtrado.columns:
                df_usar_filtrado[var_objetivo_clas] = df_usar_filtrado[var_objetivo_clas].astype(int)

            cols_present = [c for c in col_usar if c in df_usar_filtrado.columns]
            if var_objetivo in df_usar_filtrado.columns and var_objetivo not in cols_present:
                cols_present.append(var_objetivo)
            if var_objetivo_clas in df_usar_filtrado.columns and var_objetivo_clas not in cols_present:
                cols_present.append(var_objetivo_clas)

            faltantes = [c for c in col_usar if c not in df_usar_filtrado.columns]
            if faltantes:
                print(f"[Aviso] Algunas columnas de col_usar no existen y se omiten: {faltantes}")

            df_usar_filtrado = df_usar_filtrado[cols_present].copy()
            print(f"[Info] df_usar_filtrado reducido a {df_usar_filtrado.shape[1]} columnas y {len(df_usar_filtrado)} filas")

            df_usar_filtrado, num_filas_eliminadas = eliminar_filas_por_columna(df_usar_filtrado)
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            col_usar = [elemento for elemento in col_usar if elemento not in faltantes]

            num_estud_retiros = len(df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 1])
            num_total_estud = len(df_usar_filtrado)
            porc_retiros = num_estud_retiros / num_total_estud

            print(f"[Aviso] El porcentaje de retiros es  ({porc_retiros:.2%}).")
            if porc_retiros > 0.03:
                print("\n-------------------Modelo de Clasificacion de Retiros (RF)--------------------\n")
                print(col_usar)

                modelo_clasif, df_pred_clasif, metricas_clasif, X_train_clasif, X_test_clasif, y_train_clasif, y_test_clasif, df_resultados_stats_class_temp = entrenar_rf_classif(
                    df=df_usar_filtrado,
                    columna_objetivo=var_objetivo_clas,
                    columnas_predictores=col_usar,
                )

                col_prediccion_clasif = "Clasificacion_RF"

                df_resultados_stats_class_temp["Descripcion_Materia"] = nombre_asig
                df_resultados_stats_class_temp["Cod materia curso"] = asig

                if not isinstance(df_resultados_class_retiro_stats, pd.DataFrame):
                    df_resultados_class_retiro_stats = df_resultados_stats_class_temp
                else:
                    df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp], axis=0)

                df_usar_filtrado[col_prediccion_clasif] = df_pred_clasif[col_prediccion_clasif]

            print("\n-------------------Modelo de Regresion de Nota (RF)--------------------\n")

            modelo, df_pred, metricas, X_train, X_test, y_train, y_test, df_resultados_stats_temp = entrenar_rf_regresion(
                df=df_usar_filtrado,
                columna_objetivo=var_objetivo,
                columnas_predictores=col_usar,
                tuning=False,
            )

            col_prediccion = "Prediccion_RF"

            df_resultados_stats_temp["Descripcion_Materia"] = nombre_asig
            df_resultados_stats_temp["Cod materia curso"] = asig

            if not isinstance(df_resultados_stats, pd.DataFrame):
                df_resultados_stats = df_resultados_stats_temp
            else:
                df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp], axis=0)

            df_usar_filtrado[col_prediccion] = df_pred[col_prediccion]

            interpretar_rf_shap(modelo, X_test, col_usar, top_n=1, id_check=0)

            cols_to_category = cols_to_category + ["interpretacion_general", "interpretacion_registro"]
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            if df_resultados_final is None or df_resultados_final.empty:
                df_resultados_final = df_usar_filtrado
            else:
                df_resultados_final = pd.concat([df_resultados_final, df_usar_filtrado], axis=0)

        except Exception as e:
            print(f"[Error] Fallo procesando asignatura {asig}: {e}")
            continue

    return df_resultados_final, df_resultados_stats, df_resultados_class_retiro_stats


def correr_modelos_catboost_por_asignatura(df_usar):
    col_usar = []
    var_objetivo = ""

    df_resultados_final = pd.DataFrame()
    df_resultados_stats = pd.DataFrame()
    df_resultados_class_retiro_stats = pd.DataFrame()

    asig_a_usar = df_usar["Cod materia curso"].unique().tolist()

    for asig in asig_a_usar:
        try:
            print("\n" + "=" * 80)
            nombre_asig = df_usar[df_usar["Cod materia curso"] == asig]["Descripcion_Materia"].iloc[0]
            print(f"\n == Resultados para programa: {asig} - {nombre_asig} == \n")

            df_usar_filtrado = df_usar[df_usar["Cod materia curso"] == asig].copy()

            tipo_asig = None
            if "Observacion_Prerrequisito" in df_usar_filtrado.columns:
                tipo_series = df_usar_filtrado["Observacion_Prerrequisito"].dropna()
                if len(tipo_series) > 0:
                    tipo_asig = tipo_series.iloc[0]

            if tipo_asig == "Prerrequisito cumplido":
                print("[Info] Asignatura con prerequisitos. Usando logica CON PRE REQUISITOS.")
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado, tiene_prereq=True
                )
                lista_prereq_usar, df_usar_filtrado = columnas_prereq_validas_ext_dif_prereq(
                    df_usar_filtrado, df_historial_asignaturas_nombres, 0.8
                )
            else:
                print("[Info] Asignatura sin prerequisitos. Usando logica SIN PRE REQUISITOS.")
                df_usar_filtrado, col_usar, var_objetivo = renombrar_columnas(
                    df_usar_filtrado, tiene_prereq=False
                )
                lista_prereq_usar = columnas_prereq_validas(df_usar_filtrado, 0.8)

            col_usar = col_usar + lista_prereq_usar

            print(f"Columnas a usar ({len(col_usar)}): {col_usar} \n  Numero de filas a tener en cuenta: {len(df_usar_filtrado)}")

            cols_to_category = [
                "programa",
                "sexo",
                "procedencia_categoria",
                "profesor_codigo",
                "Tipo_colegio",
                "Tipo_calendario",
            ]

            var_objetivo_clas = "Retiro_Asignatura_Cat"
            if var_objetivo_clas in df_usar_filtrado.columns:
                df_usar_filtrado[var_objetivo_clas] = df_usar_filtrado[var_objetivo_clas].astype(int)

            cols_present = [c for c in col_usar if c in df_usar_filtrado.columns]
            if var_objetivo in df_usar_filtrado.columns and var_objetivo not in cols_present:
                cols_present.append(var_objetivo)
            if var_objetivo_clas in df_usar_filtrado.columns and var_objetivo_clas not in cols_present:
                cols_present.append(var_objetivo_clas)

            faltantes = [c for c in col_usar if c not in df_usar_filtrado.columns]
            if faltantes:
                print(f"[Aviso] Algunas columnas de col_usar no existen y se omiten: {faltantes}")

            df_usar_filtrado = df_usar_filtrado[cols_present].copy()
            print(f"[Info] df_usar_filtrado reducido a {df_usar_filtrado.shape[1]} columnas y {len(df_usar_filtrado)} filas")

            df_usar_filtrado, num_filas_eliminadas = eliminar_filas_por_columna(df_usar_filtrado)
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            col_usar = [elemento for elemento in col_usar if elemento not in faltantes]

            num_estud_retiros = len(df_usar_filtrado[df_usar_filtrado[var_objetivo_clas] == 1])
            num_total_estud = len(df_usar_filtrado)
            porc_retiros = num_estud_retiros / num_total_estud

            print(f"[Aviso] El porcentaje de retiros es  ({porc_retiros:.2%}).")
            if porc_retiros > 0.03:
                print("\n-------------------Modelo de Clasificacion de Retiros (CAT)--------------------\n")
                print(col_usar)

                modelo_clasif, df_pred_clasif, metricas_clasif, X_train_clasif, X_test_clasif, y_train_clasif, y_test_clasif, df_resultados_stats_class_temp = entrenar_catboost_classif(
                    df=df_usar_filtrado,
                    columna_objetivo=var_objetivo_clas,
                    columnas_predictores=col_usar,
                )

                col_prediccion_clasif = "Clasificacion_CAT"

                df_resultados_stats_class_temp["Descripcion_Materia"] = nombre_asig
                df_resultados_stats_class_temp["Cod materia curso"] = asig

                if not isinstance(df_resultados_class_retiro_stats, pd.DataFrame):
                    df_resultados_class_retiro_stats = df_resultados_stats_class_temp
                else:
                    df_resultados_class_retiro_stats = pd.concat([df_resultados_class_retiro_stats, df_resultados_stats_class_temp], axis=0)

                df_usar_filtrado[col_prediccion_clasif] = df_pred_clasif[col_prediccion_clasif]

            print("\n-------------------Modelo de Regresion de Nota (CAT)--------------------\n")

            modelo, df_pred, metricas, X_train, X_test, y_train, y_test, df_resultados_stats_temp = entrenar_catboost_regresion(
                df=df_usar_filtrado,
                columna_objetivo=var_objetivo,
                columnas_predictores=col_usar,
                tuning=False,
            )

            col_prediccion = "Prediccion_CAT"

            df_resultados_stats_temp["Descripcion_Materia"] = nombre_asig
            df_resultados_stats_temp["Cod materia curso"] = asig

            if not isinstance(df_resultados_stats, pd.DataFrame):
                df_resultados_stats = df_resultados_stats_temp
            else:
                df_resultados_stats = pd.concat([df_resultados_stats, df_resultados_stats_temp], axis=0)

            df_usar_filtrado[col_prediccion] = df_pred[col_prediccion]

            interpretar_catboost_shap(modelo, X_test, col_usar, top_n=1, id_check=0)

            cols_to_category = cols_to_category + ["interpretacion_general", "interpretacion_registro"]
            df_usar_filtrado = cambiar_a_category(df_usar_filtrado, cols_to_category)

            if df_resultados_final is None or df_resultados_final.empty:
                df_resultados_final = df_usar_filtrado
            else:
                df_resultados_final = pd.concat([df_resultados_final, df_usar_filtrado], axis=0)

        except Exception as e:
            print(f"[Error] Fallo procesando asignatura {asig}: {e}")
            continue

    return df_resultados_final, df_resultados_stats, df_resultados_class_retiro_stats

# %% rf_catboost_calls
# Ejecutar pipelines RF y CatBoost

df_resultados_final_rf, df_resultados_stats_rf, df_resultados_class_retiro_stats_rf = correr_modelos_rf_por_asignatura(df_usar)
df_resultados_final_cat, df_resultados_stats_cat, df_resultados_class_retiro_stats_cat = correr_modelos_catboost_por_asignatura(df_usar)

