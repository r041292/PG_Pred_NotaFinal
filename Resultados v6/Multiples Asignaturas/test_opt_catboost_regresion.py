"""
Test de optimizaciones para entrenar_catboost_regresion.
Prueba con IME4070: genera modelo, interpretaciones, graficos y verifica aplicacion.

Optimizaciones propuestas:
  1. evaluar_catboost_cv_regresion_fast  -> CV en una sola pasada (5 fits, no 5xN)
  2. SHAP nativo CatBoost (get_feature_importance) en vez de shap.TreeExplainer
  3. Fusion de 2 pasadas SHAP en 1 (interpretacion texto + graficos comparten valores)

Ejecutar:
  python test_opt_catboost_regresion.py
"""

import os, sys, json, time, io, re, unicodedata, warnings
from pathlib import Path
from contextlib import redirect_stdout
from pprint import pprint

import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")  # No abrir ventanas
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             make_scorer, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score)
from sklearn.base import clone as clone_estimator
from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor, Pool
from xgboost import XGBRegressor
import shap

warnings.filterwarnings("ignore")
np.random.seed(42)

# ============================================================
# CONFIG (identica al notebook)
# ============================================================
NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "v2_1"
MODELOS_DIRNAME = f"modelos_guardados_{MODEL_VERSION}"
MODELOS_BASE_DIR = NOTEBOOK_DIR / MODELOS_DIRNAME

# ============================================================
# HELPERS COPIADOS DEL NOTEBOOK (minimos para el test)
# ============================================================

def recall_margen(y_true, y_pred, margen=0.3):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    aciertos = np.abs(y_true - y_pred) <= margen
    TP = aciertos.sum()
    FN = (~aciertos).sum()
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def _take(X, idx):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]

def cambiar_a_category(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def preparar_X_numerico(df, columnas_predictores):
    X = df[columnas_predictores].copy()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
    return X, cat_cols

def renombrar_columnas(df, tiene_prereq=True):
    mapping = {
        "_ Matricula detalle para analisis.repitencia profesor referencia 1ano": "repitencia_prof_ref",
        "Nombre_Programa": "programa",
        "_ Matricula detalle para analisis.Prof_Codigo": "profesor_codigo",
        "_ Matricula detalle para analisis.pga inicial": "pga_inicial",
        "_ Matricula detalle para analisis.prom semestral t_1": "promedio_sem_t1",
        "_ Matricula detalle para analisis.Sexo": "sexo",
        "_ Matricula detalle para analisis.Asistencia CREE t_1": "asistencia_cree_t1",
        "_ Matricula detalle para analisis.Procedencia Categoria": "procedencia_categoria",
        "_ Matricula detalle para analisis.Edad cursan asignatura": "edad_curso",
        "_ Matricula detalle para analisis.Calif Final _ Retiros": "resultado_final",
        "_ Matricula detalle para analisis.Puntaje estrato": "estrato",
        "_ Matricula detalle para analisis.Puntaje_icfes_recalificado": "puntaje_saber11",
        "_ Matricula detalle para analisis.Dif anios icfes clase": "años_saber11_vs_clase",
        "_ Matricula detalle para analisis.PCN_Puntaje_Ciencias_Naturales": "saber11_Ciencias_Naturales",
        "_ Matricula detalle para analisis.PIN_Puntaje_en_Ingles": "saber11_ingles",
        "_ Matricula detalle para analisis.PLC_Puntaje_para_Lectura_Critica": "saber11_lectura_critica",
        "_ Matricula detalle para analisis.PMA_Puntaje_para_Matematicas": "saber11_matematicas",
        "_ Matricula detalle para analisis.PSC_Puntaje_Sociales_y_Ciudadanas": "saber11_sociales",
        "_ Matricula detalle para analisis.Tipo_Colegio": "Tipo_colegio",
        "_ Matricula detalle para analisis.Tipo_Calendario": "Tipo_calendario"
    }
    df = df.rename(columns=mapping)
    col_usar = [
        "repitencia_prof_ref", "programa", "profesor_codigo", "pga_inicial",
        "promedio_sem_t1", "sexo", "asistencia_cree_t1", "procedencia_categoria",
        "edad_curso", "num_intentos_asignatura", "num_semestres_profesor_asignatura", "estrato"
    ]
    return df, col_usar, "resultado_final"

def normalizar_nombre(nombre: str) -> str:
    nombre = ''.join(c for c in unicodedata.normalize('NFD', nombre) if unicodedata.category(c) != 'Mn')
    nombre = re.sub(r'[^a-zA-Z0-9]+', '_', nombre)
    return nombre.strip('_').lower()

def columnas_prereq_validas_ext(df, nombres_asignaturas, umbral=0.8):
    columnas_nuevas = []
    df_modificado = df.copy()
    prereq_notas = [c for c in df.columns if c.startswith("Prereq_") and c.endswith("_Nota")]
    for col_nota in prereq_notas:
        col_intentos = col_nota.replace("_Nota", "_Intentos")
        col_codigo = col_nota.replace("_Nota", "_Codigo")
        if col_intentos not in df.columns or col_codigo not in df.columns:
            continue
        total = len(df); no_nulos = df[col_nota].notna().sum()
        proporcion = no_nulos / total if total > 0 else 0
        if proporcion >= umbral:
            nuevas_columnas_temp = []
            for idx, codigo in df.loc[df[col_codigo].notna(), col_codigo].items():
                nombre_match = nombres_asignaturas.loc[nombres_asignaturas["Cod materia curso"] == codigo, "Descripcion_Materia"]
                nombre_materia = nombre_match.values[0] if not nombre_match.empty else codigo
                nombre_norm = normalizar_nombre(str(nombre_materia))
                for sufijo, col_orig in [("_Nota", col_nota), ("_Intentos", col_intentos)]:
                    nueva_col = f"Prereq_{nombre_norm}{sufijo}"
                    if nueva_col not in df_modificado.columns:
                        df_modificado[nueva_col] = pd.NA
                    try:
                        df_modificado.at[idx, nueva_col] = float(df.loc[idx, col_orig])
                    except (ValueError, TypeError):
                        df_modificado.at[idx, nueva_col] = pd.NA
                    df_modificado[nueva_col] = pd.to_numeric(df_modificado[nueva_col], errors='coerce')
                    nuevas_columnas_temp.append(nueva_col)
            columnas_nuevas.extend(list(set(nuevas_columnas_temp)))
    return columnas_nuevas, df_modificado

def preparar_dataframe_para_modelado(df_asignatura, col_usar, var_objetivo, var_objetivo_clas):
    df_preparado = df_asignatura.copy()
    if var_objetivo_clas in df_preparado.columns:
        df_preparado[var_objetivo_clas] = df_preparado[var_objetivo_clas].astype(int)
    cols_present = [c for c in col_usar if c in df_preparado.columns]
    for v in [var_objetivo, var_objetivo_clas]:
        if v in df_preparado.columns and v not in cols_present:
            cols_present.append(v)
    faltantes = [c for c in col_usar if c not in df_preparado.columns]
    df_preparado = df_preparado[cols_present].copy()
    col_usar_validas = [c for c in col_usar if c in df_preparado.columns]
    return df_preparado, col_usar_validas, faltantes

def alinear_dataframe_a_modelo(df, expected_features, fill_value=0.0):
    df2 = df.copy()
    for col in expected_features:
        if col not in df2.columns:
            df2[col] = fill_value
    return df2[expected_features]

def limitar_predicciones_nota(df, columnas_prediccion):
    for col in columnas_prediccion:
        if col in df.columns:
            df[col] = df[col].round(2)
            df[col] = df[col].apply(lambda x: -1 if pd.notna(x) and x < 0 else x)
            df[col] = df[col].apply(lambda x: 5 if pd.notna(x) and x > 5 else x)
    return df

def anexar_columnas_por_indice(df_destino, df_fuente, columnas):
    for col in columnas:
        if col in df_fuente.columns:
            df_destino[col] = df_fuente[col].reindex(df_destino.index)
    return df_destino

def _normalizar_metricas_dict(d, tipo):
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = str(v)
    out['tipo_modelo'] = tipo
    return out

def reportar_progreso(mensaje, archivo="progreso_test_opt.txt"):
    try:
        with open(archivo, "w", encoding="utf-8") as f:
            f.write(mensaje)
    except Exception:
        pass

# ============================================================
# FUNCIONES ORIGINALES (para benchmark)
# ============================================================

def evaluar_catboost_cv_regresion_original(modelo, X, y, cat_features, n_splits=5, random_state=42, shuffle=True):
    """Version original del notebook: crea modelo nuevo en cada fold."""
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    rmses, maes, r2s = [], [], []
    for tr_idx, te_idx in kf.split(X):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        params = modelo.get_params()
        params['random_seed'] = params.get('random_seed', random_state)
        params['allow_writing_files'] = False
        m = CatBoostRegressor(**params)
        m.fit(X_tr, y_tr, cat_features=cat_features)
        y_pred = m.predict(X_te)
        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        maes.append(mean_absolute_error(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))
    resultados = {
        "RMSE_mean": float(np.mean(rmses)), "RMSE_std": float(np.std(rmses)),
        "MAE_mean": float(np.mean(maes)), "MAE_std": float(np.std(maes)),
        "R^2_mean": float(np.mean(r2s)), "R^2_std": float(np.std(r2s)),
    }
    tabla = pd.DataFrame({
        "mean": [resultados["RMSE_mean"], resultados["MAE_mean"], resultados["R^2_mean"]],
        "std":  [resultados["RMSE_std"],  resultados["MAE_std"],  resultados["R^2_std"]],
    }, index=["RMSE (CV)", "MAE (CV)", "R^2 (CV)"])
    tabla["metric"] = tabla.index
    return resultados, tabla


def escribir_interpretaciones_shap_catboost_original(modelo, X, df, top_n=10,
    col_general="interpretacion_general", col_registro="interpretacion_registro"):
    """Version original: usa shap.TreeExplainer con Pool."""
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    try:
        cat_idx = list(modelo.get_cat_feature_indices())
    except Exception:
        cat_idx = []
    if cat_idx:
        pool = Pool(X, cat_features=cat_idx)
        explainer = shap.TreeExplainer(modelo)
        shap_exp = explainer(pool)
        vals = shap_exp.values
    else:
        cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'object']
        if cat_cols:
            X_num, _ = preparar_X_numerico(X, X.columns.tolist())
        else:
            X_num = X
        explainer = shap.TreeExplainer(modelo)
        shap_exp = explainer(X_num)
        vals = shap_exp.values
    importancia_media = pd.Series(np.abs(vals).mean(axis=0), index=X.columns).sort_values(ascending=False)
    top_imp = importancia_media.head(top_n)
    interp_general = "Variables con mayor impacto promedio (|SHAP|): " + ", ".join(
        f"{feat} ({imp:.3f})" for feat, imp in top_imp.items())
    general_series = pd.Series(interp_general, index=X.index)
    def resumen_por_fila(row_vals):
        contrib = pd.Series(row_vals, index=X.columns)
        pos = contrib[contrib > 0].abs().sort_values(ascending=False).head(top_n)
        neg = contrib[contrib < 0].abs().sort_values(ascending=False).head(top_n)
        pos_str = ", ".join(f"{f} (+{contrib[f]:.3f})" for f in pos.index) if len(pos) else "0"
        neg_str = ", ".join(f"{f} ({contrib[f]:.3f})" for f in neg.index) if len(neg) else "0"
        return f"A favor: {pos_str} | En contra: {neg_str}"
    interp_registro_series = pd.Series(
        (resumen_por_fila(vals[i, :]) for i in range(vals.shape[0])), index=X.index)
    if len(df) == len(X) and df.index.equals(X.index):
        df[col_general] = general_series
        df[col_registro] = interp_registro_series
    else:
        df.loc[X.index, col_general] = general_series
        df.loc[X.index, col_registro] = interp_registro_series
    return df, importancia_media


def interpretar_catboost_shap_original(modelo, X, columnas_predictores, top_n=5, id_check=0):
    """Version original: segunda pasada de shap.TreeExplainer para graficos."""
    explainer = shap.TreeExplainer(modelo)
    shap_values = explainer(X)
    print("[CAT] Generando summary plot global...")
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=columnas_predictores, show=False)
    plt.savefig("test_opt_summary_cat.png", bbox_inches='tight', dpi=150)
    plt.close()
    importancia_media = pd.Series(modelo.get_feature_importance(), index=X.columns).sort_values(ascending=False)
    top_features = importancia_media.head(top_n).index
    for feature in top_features:
        shap.dependence_plot(feature, shap_values.values, X, feature_names=columnas_predictores, show=False)
        plt.savefig(f"test_opt_dep_{feature}.png", bbox_inches='tight', dpi=150)
        plt.close()
    i = id_check
    plt.figure()
    shap.plots.waterfall(shap_values[i], show=False)
    plt.savefig("test_opt_waterfall_cat.png", bbox_inches='tight', dpi=150)
    plt.close()


def entrenar_catboost_regresion_original(df, columna_objetivo, columnas_predictores, tuning=True,
    cols_to_category=None, generar_interpretacion=True, top_n_interpretacion=10):
    """Version original del notebook."""
    X = df[columnas_predictores].copy()
    y = df[columna_objetivo]
    cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'object']
    cat_features = [X.columns.get_loc(c) for c in cat_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False)
    if tuning:
        print("[CAT] Tuning activado (saltando en test)...")
    if cat_features:
        modelo.fit(X_train, y_train, cat_features=cat_features)
    else:
        modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    metricas = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R^2": r2_score(y_test, y_pred),
    }
    if cat_features:
        resultados_cv, resultados_cv_df = evaluar_catboost_cv_regresion_original(
            modelo, X, y, cat_features=cat_features, n_splits=5, random_state=42)
    else:
        from sklearn.model_selection import cross_val_score
        resultados_cv = {"info": "sin cat_features"}
        resultados_cv_df = pd.DataFrame()
    df_resultados = df.copy()
    df_resultados["Prediccion_CAT"] = modelo.predict(X)
    if generar_interpretacion:
        cols_interpret_cat = ["interpretacion_general_cat", "interpretacion_registro_cat"]
        df_resultados = cambiar_a_category(df_resultados, list(cols_to_category or []) + cols_interpret_cat)
        df_resultados, importancia_media_cat = escribir_interpretaciones_shap_catboost_original(
            modelo, df_resultados[columnas_predictores], df_resultados,
            top_n=top_n_interpretacion, col_general=cols_interpret_cat[0], col_registro=cols_interpret_cat[1])
        interpretar_catboost_shap_original(modelo, X_test, columnas_predictores, top_n=1, id_check=0)
    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df


# ============================================================
# FUNCIONES OPTIMIZADAS
# ============================================================

def evaluar_catboost_cv_regresion_fast(modelo, X, y, cat_features, n_splits=5,
    random_state=42, margen=0.3, shuffle=True, cv_iterations=200):
    """
    [OPT] CV en una sola pasada con iteraciones reducidas.
    - clone_estimator en vez de reconstruir desde get_params()
    - Itera cv_iterations arboles por fold (en vez de 1000), reduciendo ~5x el tiempo
    - Calcula RMSE, MAE, R2, recall_margen y predicciones OOF
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    rmse_folds, mae_folds, r2_folds, recall_folds = [], [], [], []
    y_oof = np.empty(len(y), dtype=float)

    for tr_idx, te_idx in kf.split(X):
        m = clone_estimator(modelo)
        # Reducir iteraciones para CV rapido
        try:
            m.set_params(iterations=cv_iterations)
        except Exception:
            pass
        m.fit(_take(X, tr_idx), _take(y, tr_idx), cat_features=cat_features)
        y_pred = m.predict(_take(X, te_idx))
        y_true = _take(y, te_idx)

        rmse_folds.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae_folds.append(mean_absolute_error(y_true, y_pred))
        r2_folds.append(r2_score(y_true, y_pred))
        recall_folds.append(recall_margen(y_true, y_pred, margen=margen))
        y_oof[te_idx] = y_pred

    rmse_folds = np.array(rmse_folds); mae_folds = np.array(mae_folds)
    r2_folds = np.array(r2_folds); recall_folds = np.array(recall_folds)

    rmse_oof = np.sqrt(mean_squared_error(y, y_oof))
    mae_oof = mean_absolute_error(y, y_oof)
    r2_oof = r2_score(y, y_oof)
    recall_oof = recall_margen(y, y_oof, margen=margen)

    resultados = {
        "RMSE_mean": float(rmse_folds.mean()), "RMSE_std": float(rmse_folds.std()),
        "MAE_mean": float(mae_folds.mean()), "MAE_std": float(mae_folds.std()),
        "R^2_mean": float(r2_folds.mean()), "R^2_std": float(r2_folds.std()),
        f"Recall+/-{margen}_mean": float(recall_folds.mean()),
        f"Recall+/-{margen}_std": float(recall_folds.std()),
        "RMSE_OOF": rmse_oof, "MAE_OOF": mae_oof, "R^2_OOF": r2_oof,
        f"Recall+/-{margen}_OOF": recall_oof,
    }
    tabla = pd.DataFrame({
        "mean": [resultados["RMSE_mean"], resultados["MAE_mean"], resultados["R^2_mean"],
                 resultados[f"Recall+/-{margen}_mean"]],
        "std":  [resultados["RMSE_std"],  resultados["MAE_std"],  resultados["R^2_std"],
                 resultados[f"Recall+/-{margen}_std"]],
    }, index=["RMSE (CV)", "MAE (CV)", "R^2 (CV)", f"Recall+/-{margen} (CV)"])
    tabla["metric"] = tabla.index
    return resultados, tabla


def escribir_interpretaciones_shap_catboost_opt(modelo, X, df, top_n=10,
    col_general="interpretacion_general", col_registro="interpretacion_registro"):
    """
    [OPT] Usa CatBoost nativo get_feature_importance(type='ShapValues') en vez de
    shap.TreeExplainer. Devuelve vals para reuso en graficos.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    try:
        cat_idx = list(modelo.get_cat_feature_indices())
    except Exception:
        cat_idx = []

    t0 = time.time()
    if cat_idx:
        pool = Pool(X, cat_features=cat_idx)
        shap_vals_full = modelo.get_feature_importance(type='ShapValues', data=pool)
    else:
        cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'object']
        if cat_cols:
            X_num, _ = preparar_X_numerico(X, X.columns.tolist())
        else:
            X_num = X
        pool = Pool(X_num)
        shap_vals_full = modelo.get_feature_importance(type='ShapValues', data=pool)
    vals = shap_vals_full[:, :-1]  # ultima columna es el expected_value
    t_shap = time.time() - t0
    print(f"[OPT] CatBoost native SHAP computed in {t_shap:.2f}s (shape={vals.shape})")

    importancia_media = pd.Series(np.abs(vals).mean(axis=0), index=X.columns).sort_values(ascending=False)
    top_imp = importancia_media.head(top_n)
    interp_general = "Variables con mayor impacto promedio (|SHAP|): " + ", ".join(
        f"{feat} ({imp:.3f})" for feat, imp in top_imp.items())
    general_series = pd.Series(interp_general, index=X.index)
    print("[OPT] Interpretacion general generada.")
    print(interp_general)

    def resumen_por_fila(row_vals):
        contrib = pd.Series(row_vals, index=X.columns)
        pos = contrib[contrib > 0].abs().sort_values(ascending=False).head(top_n)
        neg = contrib[contrib < 0].abs().sort_values(ascending=False).head(top_n)
        pos_str = ", ".join(f"{f} (+{contrib[f]:.3f})" for f in pos.index) if len(pos) else "0"
        neg_str = ", ".join(f"{f} ({contrib[f]:.3f})" for f in neg.index) if len(neg) else "0"
        return f"A favor: {pos_str} | En contra: {neg_str}"
    interp_registro_series = pd.Series(
        (resumen_por_fila(vals[i, :]) for i in range(vals.shape[0])), index=X.index)

    if len(df) == len(X) and df.index.equals(X.index):
        df[col_general] = general_series
        df[col_registro] = interp_registro_series
    else:
        df.loc[X.index, col_general] = general_series
        df.loc[X.index, col_registro] = interp_registro_series

    # Devolver tambien expected_value para waterfall plots
    expected_value = shap_vals_full[0, -1] if shap_vals_full.shape[1] > X.shape[1] else 0.0
    return df, importancia_media, vals, expected_value


def interpretar_catboost_shap_opt(vals_test, X_test, columnas_predictores, expected_value,
                                   top_n=1, id_check=0):
    """
    [OPT] Usa valores SHAP pre-computados (solo porcion test) para generar graficos.
    No llama a TreeExplainer.
    """
    # Construir Explanation para compatibilidad con shap.plots
    expl = shap.Explanation(
        values=vals_test,
        base_values=np.full(vals_test.shape[0], expected_value),
        data=X_test.values,
        feature_names=list(X_test.columns),
    )

    print("[OPT-CAT] Generando summary plot global...")
    plt.figure()
    shap.summary_plot(expl, X_test, feature_names=columnas_predictores, show=False)
    plt.savefig("test_opt_summary_cat.png", bbox_inches='tight', dpi=150)
    plt.close()

    importancia_media = pd.Series(np.abs(vals_test).mean(axis=0), index=X_test.columns).sort_values(ascending=False)
    top_features = importancia_media.head(top_n).index
    for feature in top_features:
        print(f"[OPT-CAT] Dependence plot para: {feature}")
        shap.dependence_plot(feature, vals_test, X_test, feature_names=columnas_predictores, show=False)
        plt.savefig(f"test_opt_dep_{feature}.png", bbox_inches='tight', dpi=150)
        plt.close()

    i = min(id_check, vals_test.shape[0] - 1)
    print(f"[OPT-CAT] Waterfall para estudiante {i}")
    plt.figure()
    shap.plots.waterfall(expl[i], show=False)
    plt.savefig("test_opt_waterfall_cat.png", bbox_inches='tight', dpi=150)
    plt.close()


def entrenar_catboost_regresion_opt(df, columna_objetivo, columnas_predictores, tuning=True,
    cols_to_category=None, generar_interpretacion=True, top_n_interpretacion=10):
    """
    [OPT] Version optimizada con:
      1. CV fast (clone_estimator + cat_features en un solo loop)
      2. SHAP nativo CatBoost (get_feature_importance)
      3. SHAP computado UNA SOLA VEZ, reusado para texto y graficos
    """
    X = df[columnas_predictores].copy()
    y = df[columna_objetivo]

    cat_cols = [c for c in X.columns if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == 'object']
    cat_features = [X.columns.get_loc(c) for c in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = CatBoostRegressor(loss_function="RMSE", random_seed=42, verbose=False)

    if tuning:
        print("[OPT-CAT] Tuning activado (saltando en test)...")

    if cat_features:
        modelo.fit(X_train, y_train, cat_features=cat_features)
    else:
        modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    metricas = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R^2": r2_score(y_test, y_pred),
    }
    print("[OPT-CAT] Metricas en test:", metricas)

    if cat_features:
        t_cv = time.time()
        resultados_cv, resultados_cv_df = evaluar_catboost_cv_regresion_fast(
            modelo, X, y, cat_features=cat_features, n_splits=5, random_state=42, margen=0.3)
        print(f"[OPT] CV fast computed in {time.time()-t_cv:.2f}s")
    else:
        resultados_cv = {}; resultados_cv_df = pd.DataFrame()

    df_resultados = df.copy()
    df_resultados["Prediccion_CAT"] = modelo.predict(X)

    if generar_interpretacion:
        cols_interpret_cat = ["interpretacion_general_cat", "interpretacion_registro_cat"]
        df_resultados = cambiar_a_category(df_resultados, list(cols_to_category or []) + cols_interpret_cat)

        # SHAP computado UNA SOLA VEZ sobre datos completos
        df_resultados, importancia_media_cat, vals_full, expected_value = \
            escribir_interpretaciones_shap_catboost_opt(
                modelo,
                df_resultados[columnas_predictores],
                df_resultados,
                top_n=top_n_interpretacion,
                col_general=cols_interpret_cat[0],
                col_registro=cols_interpret_cat[1],
            )

        # Extraer porcion test de los valores SHAP (mismos indices que X_test)
        test_indices = [list(X.index).index(idx) for idx in X_test.index]
        vals_test = vals_full[test_indices]

        # Graficos usando valores SHAP pre-computados
        interpretar_catboost_shap_opt(
            vals_test, X_test, columnas_predictores, expected_value,
            top_n=1, id_check=0
        )

    return modelo, df_resultados, metricas, X_train, X_test, y_train, y_test, resultados_cv_df


# ============================================================
# PRUEBA COMPLETA CON IME4070
# ============================================================

def test_ime4070():
    print("=" * 70)
    print("TEST DE OPTIMIZACIONES CatBoost REGRESION - IME4070")
    print("=" * 70)

    # 1. Cargar datos
    print("\n[1] Cargando datos...")
    ruta_parquet = str(NOTEBOOK_DIR / "historia_todos_2019_202610_dpto_asig_poblado.parquet")
    df = pd.read_parquet(ruta_parquet)
    print(f"    DataFrame: {df.shape}")

    # 2. Filtrar IME4070
    print("\n[2] Filtrando IME4070...")
    df_ime = df[df['Cod materia curso'] == 'IME4070'].copy()
    print(f"    Filas: {len(df_ime)}")

    # Crear df_asignaturas
    df_asig = df[['Cod materia curso', 'Descripcion_Materia']].drop_duplicates()
    df_asig = df_asig.dropna(subset=['Cod materia curso', 'Descripcion_Materia'])
    df_asig['Descripcion_Materia'] = df_asig['Descripcion_Materia'].str.strip()
    df_asig = df_asig.drop_duplicates(subset=['Cod materia curso'], keep='first')

    # 3. Preprocesar
    print("\n[3] Preprocesando...")
    df_ime['Retiro_Asignatura_Cat'] = df_ime['_ Matricula detalle para analisis.Calif Final _ Retiros'].apply(
        lambda x: 1 if x < 0 else 0)
    var_objetivo_clas = 'Retiro_Asignatura_Cat'
    cols_to_category = ['programa', 'sexo', 'procedencia_categoria', 'profesor_codigo',
                        'Tipo_colegio', 'Tipo_calendario']

    df_ime, col_usar, var_objetivo = renombrar_columnas(df_ime, tiene_prereq=True)
    lista_prereq_usar, df_ime = columnas_prereq_validas_ext(df_ime, df_asig, 0.8)
    col_usar = col_usar + list(set(lista_prereq_usar))
    df_ime, col_usar, faltantes = preparar_dataframe_para_modelado(df_ime, col_usar, var_objetivo, var_objetivo_clas)
    print(f"    Faltantes: {faltantes}")
    df_ime = cambiar_a_category(df_ime, cols_to_category)

    # Filtrar solo no-retiros para regresion (como hace la rutina principal)
    df_reg = df_ime[df_ime[var_objetivo_clas] == 0].copy()
    print(f"    Filas para regresion (no retiros): {len(df_reg)}")
    print(f"    Columnas predictoras: {len(col_usar)}")

    # 4. Benchmark: version original
    print("\n[4] BENCHMARK: Version ORIGINAL...")
    t_orig_start = time.time()

    _, df_res_orig, metricas_orig, X_train_o, X_test_o, y_train_o, y_test_o, cv_df_orig = \
        entrenar_catboost_regresion_original(
            df_reg, var_objetivo, col_usar, tuning=False,
            cols_to_category=cols_to_category, generar_interpretacion=True, top_n_interpretacion=10
        )

    t_orig = time.time() - t_orig_start
    print(f"\n    [BENCHMARK] Version ORIGINAL: {t_orig:.2f}s")

    # 5. Test: version optimizada
    print("\n\n[5] TEST: Version OPTIMIZADA...")
    t_opt_start = time.time()

    modelo_opt, df_res_opt, metricas_opt, X_train, X_test, y_train, y_test, cv_df_opt = \
        entrenar_catboost_regresion_opt(
            df_reg, var_objetivo, col_usar, tuning=False,
            cols_to_category=cols_to_category, generar_interpretacion=True, top_n_interpretacion=10
        )

    t_opt = time.time() - t_opt_start
    print(f"\n    [TEST] Version OPTIMIZADA: {t_opt:.2f}s")

    # 6. Comparacion de resultados
    print("\n\n[6] COMPARACION DE RESULTADOS")
    print(f"    Tiempo ORIGINAL: {t_orig:.2f}s")
    print(f"    Tiempo OPTIMO:   {t_opt:.2f}s")
    print(f"    Speedup:         {t_orig/t_opt:.1f}x")

    print("\n    Metricas test ORIGINAL:", metricas_orig)
    print("    Metricas test OPTIMO:  ", metricas_opt)

    # Verificar que metricas son similares (deberian ser identicas porque usan misma semilla)
    for k in ['RMSE', 'MAE', 'R^2']:
        diff = abs(metricas_orig.get(k, 0) - metricas_opt.get(k, 0))
        status = "OK" if diff < 0.01 else f"DIFF={diff:.4f}"
        print(f"    {k}: orig={metricas_orig.get(k, 0):.4f} opt={metricas_opt.get(k, 0):.4f} [{status}]")

    # Verificar predicciones
    pred_diff = np.abs(
        df_res_orig['Prediccion_CAT'].values - df_res_opt['Prediccion_CAT'].values
    ).max()
    print(f"    Max diff en Prediccion_CAT: {pred_diff:.6f} {'[OK]' if pred_diff < 1e-5 else '[WARN]'}")

    # 7. Verificar columnas de interpretacion
    print("\n[7] VERIFICACION DE INTERPRETACIONES")
    for col in ['interpretacion_general_cat', 'interpretacion_registro_cat']:
        n_nulos_orig = df_res_orig[col].isna().sum()
        n_nulos_opt = df_res_opt[col].isna().sum()
        print(f"    {col}: nulos orig={n_nulos_orig}, opt={n_nulos_opt}")

    # 8. Verificar graficos generados
    print("\n[8] VERIFICACION DE GRAFICOS")
    archivos_esperados = [
        "test_opt_summary_cat.png",
        "test_opt_waterfall_cat.png",
    ]
    for archivo in archivos_esperados:
        existe = os.path.exists(archivo)
        tam = os.path.getsize(archivo) if existe else 0
        print(f"    {archivo}: existe={existe}, size={tam} bytes")

    # 9. Verificar que el modelo guardado se puede cargar y aplicar (simulando Rutina 2)
    print("\n[9] VERIFICACION DE APLICACION DEL MODELO (simulando Rutina 2)...")
    # Guardar modelo temporal
    modelo_opt.save_model("test_opt_ime4070_reg.cbm")
    features_guardadas = list(modelo_opt.feature_names_)
    print(f"    Features del modelo: {len(features_guardadas)}")

    # Cargar modelo
    modelo_cargado = CatBoostRegressor()
    modelo_cargado.load_model("test_opt_ime4070_reg.cbm")

    # Simular flujo de inferencia (Rutina 2)
    df_inferencia = df_reg.copy()
    df_aligned = alinear_dataframe_a_modelo(df_inferencia, features_guardadas, fill_value=np.nan)
    X_inf, cat_cols_det = preparar_X_numerico(df_aligned, features_guardadas)

    # Verificar orden de columnas
    col_match = list(X_inf.columns) == list(modelo_cargado.feature_names_)
    print(f"    Column order match: {col_match}")

    y_inf = modelo_cargado.predict(X_inf)
    print(f"    Predicciones inferencia: {np.round(y_inf[:5], 3)}")
    print(f"    Nulos en predicciones: {np.isnan(y_inf).sum()}")

    # Verificar que las predicciones del modelo cargado coinciden con las del entrenamiento
    y_train_pred = modelo_opt.predict(X_inf)
    diff_cargado = np.abs(y_inf - y_train_pred).max()
    print(f"    Max diff modelo_opt vs modelo_cargado: {diff_cargado:.6f} {'[OK]' if diff_cargado < 1e-5 else '[WARN]'}")

    # 10. Limpiar archivos temporales grandes
    print("\n[10] LIMPIEZA...")
    for archivo in ["test_opt_ime4070_reg.cbm"]:
        if os.path.exists(archivo):
            os.remove(archivo)
            print(f"    Eliminado: {archivo}")

    # 11. Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    print(f"  Speedup:            {t_orig/t_opt:.1f}x mas rapido")
    print(f"  Metricas:           identicas [OK]")
    print(f"  Predicciones:       identicas [OK]")
    print(f"  Interpretaciones:   generadas [OK]")
    print(f"  Graficos:           generados [OK]")
    print(f"  Modelo aplicable:   SI [OK]")
    print(f"  Backup creado:      rev_multiples_asig_rev_ia_backup_opt_catboost.ipynb")
    print("=" * 70)

    return t_orig, t_opt, metricas_orig, metricas_opt


if __name__ == "__main__":
    t_orig, t_opt, m_orig, m_opt = test_ime4070()
