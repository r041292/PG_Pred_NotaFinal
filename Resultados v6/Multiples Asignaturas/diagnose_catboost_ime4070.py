"""
Diagnostico: CatBoost IME4070 - Kernel crash en inferencia.
Ejecutar desde el directorio del notebook:
  python diagnose_catboost_ime4070.py
"""

import os
import sys
import json
import time
import traceback
import warnings
import re
import unicodedata

import pandas as pd
import numpy as np
from pathlib import Path

from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings("ignore")

# === CONFIG (replicada del notebook) ===
NOTEBOOK_DIR = Path(__file__).resolve().parent
MODEL_VERSION = "v2_1"
MODELOS_DIRNAME = f"modelos_guardados_{MODEL_VERSION}"
MODELOS_BASE_DIR = NOTEBOOK_DIR / MODELOS_DIRNAME
LEGACY_MODELOS_BASE_DIR = NOTEBOOK_DIR / "modelos_guardados_v2"

VALID_MODELOS = ['xgboost', 'rf', 'catboost']
VALID_VARIANTS = ['main']
VALID_TASKS = ['prediccion_nota', 'prediccion_retiro']
MODEL_FOLDER_NAMES = {
    'xgboost': 'xgboost',
    'rf': 'rf',
    'catboost': 'catboost',
}

# === HELPERS MINIMOS (copiados del notebook para ser autosuficientes) ===

def normalizar_nombre(nombre: str) -> str:
    nombre = ''.join(c for c in unicodedata.normalize('NFD', nombre) if unicodedata.category(c) != 'Mn')
    nombre = re.sub(r'[^a-zA-Z0-9]+', '_', nombre)
    nombre = nombre.strip("_").lower()
    return nombre


def get_model_dir(modelo, variant, task):
    if modelo not in VALID_MODELOS:
        raise ValueError(f'Modelo invalido: {modelo}')
    return os.path.join(str(MODELOS_BASE_DIR), MODEL_FOLDER_NAMES[modelo], variant, task)


def get_model_paths(asignatura, modelo, variant, task):
    base_dir = get_model_dir(modelo, variant, task)
    ext_map = {'xgboost': '.json', 'rf': '.joblib', 'catboost': '.cbm'}
    ext = ext_map.get(modelo, '')
    nombre = str(asignatura)
    model_path = os.path.join(base_dir, f'{nombre}{ext}')
    features_path = os.path.join(base_dir, f'{nombre}_features.json')
    meta_path = os.path.join(base_dir, f'{nombre}_meta.json')
    return {'model': model_path, 'features': features_path, 'meta': meta_path}


def _legacy_model_dir(modelo, task):
    if modelo == 'xgboost':
        return os.path.join(str(LEGACY_MODELOS_BASE_DIR), 'prediccion_nota') if task == 'prediccion_nota' else os.path.join(str(LEGACY_MODELOS_BASE_DIR), 'prediccion_retiro')
    if modelo == 'rf':
        return os.path.join(str(LEGACY_MODELOS_BASE_DIR), 'prediccion_nota_rf') if task == 'prediccion_nota' else os.path.join(str(LEGACY_MODELOS_BASE_DIR), 'prediccion_retiro_rf')
    if modelo == 'catboost':
        return os.path.join(str(LEGACY_MODELOS_BASE_DIR), 'prediccion_nota_cat') if task == 'prediccion_nota' else os.path.join(str(LEGACY_MODELOS_BASE_DIR), 'prediccion_retiro_cat')
    return None


def _find_legacy_model_file(asignatura, modelo, task):
    legacy_dir = _legacy_model_dir(modelo, task)
    if legacy_dir is None or not os.path.isdir(legacy_dir):
        return None
    exts = ('.cbm',) if modelo == 'catboost' else (('.json', '.model') if modelo == 'xgboost' else ('.joblib',))
    files = [f for f in os.listdir(legacy_dir) if str(asignatura) in f and f.endswith(exts)]
    return os.path.join(legacy_dir, files[0]) if files else None


def cargar_modelo_catboost_regresion(codigo_asignatura, variant='main'):
    paths = get_model_paths(codigo_asignatura, 'catboost', variant, 'prediccion_nota')
    ruta_modelo = paths['model']
    if not os.path.isfile(ruta_modelo):
        legacy_path = _find_legacy_model_file(codigo_asignatura, 'catboost', 'prediccion_nota')
        if legacy_path is not None:
            ruta_modelo = legacy_path
        else:
            print(f'[Aviso] No se encontro modelo CatBoost de regresion para {codigo_asignatura}')
            return None
    modelo = CatBoostRegressor()
    modelo.load_model(ruta_modelo)
    print(f'[Info] Modelo de regresion CatBoost cargado desde {ruta_modelo}')
    return modelo


def cargar_modelo_catboost_clasificacion(codigo_asignatura, variant='main'):
    paths = get_model_paths(codigo_asignatura, 'catboost', variant, 'prediccion_retiro')
    ruta_modelo = paths['model']
    if not os.path.isfile(ruta_modelo):
        legacy_path = _find_legacy_model_file(codigo_asignatura, 'catboost', 'prediccion_retiro')
        if legacy_path is not None:
            ruta_modelo = legacy_path
        else:
            print(f'[Aviso] No se encontro modelo CatBoost de clasificacion para {codigo_asignatura}')
            return None
    modelo = CatBoostClassifier()
    modelo.load_model(ruta_modelo)
    print(f'[Info] Modelo de clasificacion CatBoost cargado desde {ruta_modelo}')
    return modelo


def _infer_features_from_model(modelo, modelo_nombre):
    if modelo is None:
        return None
    if modelo_nombre == 'catboost':
        try:
            feats = modelo.get_feature_names()
            if feats:
                return list(feats)
        except Exception:
            return None
    return None


def cargar_features_modelo(asignatura, modelo, variant, task, modelo_cargado=None):
    paths = get_model_paths(asignatura, modelo, variant, task)
    if os.path.isfile(paths['features']):
        with open(paths['features'], 'r', encoding='utf-8') as f:
            return json.load(f)
    inferred = _infer_features_from_model(modelo_cargado, modelo)
    return inferred


def alinear_dataframe_a_modelo(df, expected_features, fill_value=0.0):
    df2 = df.copy()
    for col in expected_features:
        if col not in df2.columns:
            df2[col] = fill_value
    df2 = df2[expected_features]
    return df2


def preparar_X_numerico(df, columnas_predictores):
    X = df[columnas_predictores].copy()
    cat_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category').cat.codes
    return X, cat_cols


def cambiar_a_category(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def preparar_dataframe_para_modelado(df_asignatura, col_usar, var_objetivo, var_objetivo_clas):
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
        print(f'[Aviso] Columnas faltantes: {faltantes}')
    df_preparado = df_preparado[cols_present].copy()
    col_usar_validas = [c for c in col_usar if c in df_preparado.columns]
    return df_preparado, col_usar_validas, faltantes


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
    if tiene_prereq:
        col_usar = [
            "repitencia_prof_ref", "programa", "profesor_codigo",
            "pga_inicial", "promedio_sem_t1", "sexo", "asistencia_cree_t1",
            "procedencia_categoria", "edad_curso", "num_intentos_asignatura",
            "num_semestres_profesor_asignatura", "estrato"
        ]
    else:
        col_usar = [
            "repitencia_prof_ref", "programa", "profesor_codigo",
            "sexo", "asistencia_cree_t1", "procedencia_categoria",
            "edad_curso", "num_intentos_asignatura",
            "num_semestres_profesor_asignatura", "estrato",
            "Tipo_colegio", "Tipo_calendario"
        ]
    var_objetivo = "resultado_final"
    return df, col_usar, var_objetivo


def columnas_prereq_validas(df, umbral=0.8):
    columnas_seleccionadas = []
    prereq_notas = [c for c in df.columns if c.startswith("Prereq_") and c.endswith("_Nota")]
    for col_nota in prereq_notas:
        col_intentos = col_nota.replace("_Nota", "_Intentos")
        if col_intentos not in df.columns:
            continue
        total = len(df)
        no_nulos = df[col_nota].notna().sum()
        proporcion = no_nulos / total if total > 0 else 0
        if proporcion >= umbral:
            columnas_seleccionadas.extend([col_nota, col_intentos])
    return columnas_seleccionadas


def columnas_prereq_validas_ext(df, nombres_asignaturas, umbral=0.8):
    columnas_nuevas = []
    df_modificado = df.copy()
    prereq_notas = [c for c in df.columns if c.startswith("Prereq_") and c.endswith("_Nota")]
    for col_nota in prereq_notas:
        col_intentos = col_nota.replace("_Nota", "_Intentos")
        col_codigo = col_nota.replace("_Nota", "_Codigo")
        if col_intentos not in df.columns or col_codigo not in df.columns:
            continue
        total = len(df)
        no_nulos = df[col_nota].notna().sum()
        proporcion = no_nulos / total if total > 0 else 0
        if proporcion >= umbral:
            nuevas_columnas_temp = []
            for idx, codigo in df.loc[df[col_codigo].notna(), col_codigo].items():
                nombre_match = nombres_asignaturas.loc[nombres_asignaturas["Cod materia curso"] == codigo, "Descripcion_Materia"]
                if not nombre_match.empty:
                    nombre_materia = nombre_match.values[0]
                else:
                    nombre_materia = codigo
                nombre_materia_norm = normalizar_nombre(str(nombre_materia))
                nueva_columna = f"Prereq_{nombre_materia_norm}_Nota"
                if nueva_columna not in df_modificado.columns:
                    df_modificado[nueva_columna] = pd.NA
                valor_nota = df.loc[idx, col_nota]
                try:
                    valor_nota_float = float(valor_nota)
                except (ValueError, TypeError):
                    valor_nota_float = pd.NA
                df_modificado.at[idx, nueva_columna] = valor_nota_float
                df_modificado[nueva_columna] = pd.to_numeric(df_modificado[nueva_columna], errors='coerce')
                nuevas_columnas_temp.append(nueva_columna)
                nueva_columna_intentos = f"Prereq_{nombre_materia_norm}_Intentos"
                if nueva_columna_intentos not in df_modificado.columns:
                    df_modificado[nueva_columna_intentos] = pd.NA
                valor_intentos = df.loc[idx, col_intentos]
                try:
                    valor_intentos_float = float(valor_intentos)
                except (ValueError, TypeError):
                    valor_intentos_float = pd.NA
                df_modificado.at[idx, nueva_columna_intentos] = valor_intentos_float
                df_modificado[nueva_columna_intentos] = pd.to_numeric(df_modificado[nueva_columna_intentos], errors='coerce')
                nuevas_columnas_temp.append(nueva_columna_intentos)
            columnas_nuevas.extend(list(set(nuevas_columnas_temp)))
    return columnas_nuevas, df_modificado


# === MAIN DIAGNOSTIC ===

print("=" * 80)
print("DIAGNOSTICO: CatBoost IME4070 - Kernel Crash")
print("=" * 80)

# 1. Cargar datos
print("\n[PASO 1] Cargando datos...")
ruta_parquet = str(NOTEBOOK_DIR / "historia_todos_2019_202610_dpto_asig_poblado.parquet")
print(f"  Ruta parquet: {ruta_parquet}")
print(f"  Existe: {os.path.exists(ruta_parquet)}")

if not os.path.exists(ruta_parquet):
    print("  ERROR: No se encontro el parquet. Probando alternativa...")
    ruta_parquet = str(NOTEBOOK_DIR / "historia_todos.parquet")
    print(f"  Ruta alternativa: {ruta_parquet}")
    print(f"  Existe: {os.path.exists(ruta_parquet)}")

if not os.path.exists(ruta_parquet):
    print("  ERROR: No se encontro ningun parquet. Abortando.")
    sys.exit(1)

df = pd.read_parquet(ruta_parquet)
print(f"  DataFrame cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

# 2. Filtrar IME4070
print("\n[PASO 2] Filtrando IME4070...")
df_ime = df[df['Cod materia curso'] == 'IME4070'].copy()
print(f"  Filas para IME4070: {len(df_ime)}")

if len(df_ime) == 0:
    print("  ERROR: No hay datos para IME4070.")
    sys.exit(1)

# Ver distribucion de Observacion_Prerrequisito
print(f"\n  Distribucion Observacion_Prerrequisito:")
print(df_ime['Observacion_Prerrequisito'].value_counts(dropna=False))

tipo_prereq = df_ime['Observacion_Prerrequisito'].iloc[0]
print(f"  Tipo prerrequisito (primer registro): {tipo_prereq}")

# 3. Crear df de nombres de asignaturas
print("\n[PASO 3] Creando df de nombres de asignaturas...")
df_asignaturas = df[['Cod materia curso', 'Descripcion_Materia']].drop_duplicates()
df_asignaturas = df_asignaturas.dropna(subset=['Cod materia curso', 'Descripcion_Materia'])
df_asignaturas['Descripcion_Materia'] = df_asignaturas['Descripcion_Materia'].str.strip()
df_asignaturas = df_asignaturas.drop_duplicates(subset=['Cod materia curso'], keep='first')
print(f"  Asignaturas unicas: {len(df_asignaturas)}")

# 4. Renombrar columnas y preparar features
print("\n[PASO 4] Preparando features...")
var_objetivo_clas = 'Retiro_Asignatura_Cat'

cols_to_category = [
    'programa', 'sexo', 'procedencia_categoria',
    'profesor_codigo', 'Tipo_colegio', 'Tipo_calendario'
]

if 'Retiro_Asignatura_Cat' not in df_ime.columns:
    print("  CREANDO Retiro_Asignatura_Cat...")
    col_resultado = "_ Matricula detalle para analisis.Calif Final _ Retiros"
    if col_resultado in df_ime.columns:
        df_ime['Retiro_Asignatura_Cat'] = df_ime[col_resultado].apply(lambda x: 1 if x < 0 else 0)
    else:
        print("  ERROR: No se encontro columna de resultado final.")
        sys.exit(1)

if tipo_prereq == 'Prerrequisito cumplido':
    df_ime, col_usar, var_objetivo = renombrar_columnas(df_ime, tiene_prereq=True)
    lista_prereq_usar, df_ime = columnas_prereq_validas_ext(df_ime, df_asignaturas, 0.8)
    print("  Tipo: CON prerrequisitos")
else:
    df_ime, col_usar, var_objetivo = renombrar_columnas(df_ime, tiene_prereq=False)
    lista_prereq_usar = columnas_prereq_validas(df_ime, 0.8)
    print("  Tipo: SIN prerrequisitos")

col_usar = col_usar + lista_prereq_usar
print(f"  col_usar ({len(col_usar)}): {col_usar[:5]}...")
print(f"  Numero de filas: {len(df_ime)}")

df_ime, col_usar, faltantes = preparar_dataframe_para_modelado(df_ime, col_usar, var_objetivo, var_objetivo_clas)
print(f"  Columnas validas: {len(col_usar)}")
print(f"  Faltantes: {faltantes}")

df_ime = cambiar_a_category(df_ime, cols_to_category)
print(f"  DataFrame final: {df_ime.shape}")
print(f"  Columnas: {list(df_ime.columns)[:10]}...")

# 5. Cargar modelos CatBoost
print("\n[PASO 5] Cargando modelos CatBoost...")
print("\n  --- Clasificacion ---")
try:
    modelo_cls = cargar_modelo_catboost_clasificacion('IME4070', variant='main')
except Exception as e:
    print(f"  ERROR cargando clasificacion: {e}")
    traceback.print_exc()
    modelo_cls = None

print("\n  --- Regresion ---")
try:
    modelo_reg = cargar_modelo_catboost_regresion('IME4070', variant='main')
except Exception as e:
    print(f"  ERROR cargando regresion: {e}")
    traceback.print_exc()
    modelo_reg = None

# 6. Inspeccionar modelos
print("\n[PASO 6] Inspeccionando modelos...")

for nombre, modelo in [('CLASIFICACION', modelo_cls), ('REGRESION', modelo_reg)]:
    if modelo is None:
        print(f"\n  {nombre}: modelo es None, saltando...")
        continue
    print(f"\n  --- {nombre} ---")
    try:
        feats = modelo.get_feature_names()
        print(f"  get_feature_names() ({len(feats)}): {feats}")
    except Exception as e:
        print(f"  get_feature_names() ERROR: {e}")

    try:
        cat_idx = modelo.get_cat_feature_indices()
        print(f"  get_cat_feature_indices(): {cat_idx}")
        if cat_idx and feats:
            for i in cat_idx:
                print(f"    Categorica idx={i}: {feats[i] if i < len(feats) else '???'}")
    except Exception as e:
        print(f"  get_cat_feature_indices() ERROR: {e}")

    try:
        params = modelo.get_all_params()
        print(f"  Parametros relevantes: loss_function={params.get('loss_function', '?')}, "
              f"random_seed={params.get('random_seed', '?')}")
    except Exception as e:
        print(f"  get_all_params() ERROR: {e}")

# 7. Resolver features (replicando _resolver_features)
print("\n[PASO 7] Resolviendo features...")

fallback_cols = df_ime.columns.tolist()

for task_name, modelo, task in [
    ('CLASIFICACION', modelo_cls, 'prediccion_retiro'),
    ('REGRESION', modelo_reg, 'prediccion_nota')
]:
    print(f"\n  --- {task_name} ({task}) ---")
    if modelo is None:
        print("  Modelo es None, saltando...")
        continue

    # Replicar _resolver_features
    features = cargar_features_modelo('IME4070', 'catboost', 'main', task, modelo_cargado=modelo)
    print(f"  cargar_features_modelo() -> {features is not None} (len={len(features) if features else 0})")

    if not features:
        features = _infer_features_from_model(modelo, 'catboost')
        print(f"  _infer_features_from_model() -> {features is not None} (len={len(features) if features else 0})")

    if not features:
        excluir = {var_objetivo, var_objetivo_clas}
        features = [c for c in fallback_cols if c not in excluir]
        print(f"  Fallback a todas las columnas (len={len(features)})")

    if features:
        features = [c for c in features if c not in {var_objetivo, var_objetivo_clas}]
        features = list(dict.fromkeys(features))
        print(f"  Features finales ({len(features)}):")
        for i, f in enumerate(features):
            en_df = f in df_ime.columns
            dtype = df_ime[f].dtype if en_df else 'N/A'
            print(f"    [{i:3d}] {f}  (en_df={en_df}, dtype={dtype})")

    # 8. Alinear y probar prediccion
    print(f"\n  [PASO 8] Probando alineacion y prediccion...")
    if features:
        df_aligned = alinear_dataframe_a_modelo(df_ime, features, fill_value=np.nan)
        print(f"  df_aligned shape: {df_aligned.shape}")
        print(f"  df_aligned dtypes: {dict(df_aligned.dtypes.value_counts())}")

        X_num, cat_cols = preparar_X_numerico(df_aligned, features)
        print(f"  X_num shape: {X_num.shape}")
        print(f"  X_num dtypes: {dict(X_num.dtypes.value_counts())}")
        print(f"  Categoricas detectadas por preparar_X_numerico: {cat_cols}")

        # Verificar que los nombres de columna coinciden
        model_feat_names = modelo.get_feature_names()
        print(f"\n  Columnas del modelo ({len(model_feat_names)}): {model_feat_names[:5]}...")
        print(f"  Columnas de X_num  ({len(X_num.columns)}): {list(X_num.columns)[:5]}...")

        if list(X_num.columns) != list(model_feat_names):
            print("  [ALERTA] ORDEN DE COLUMNAS DIFERENTE!")
            print(f"  Modelo espera: {model_feat_names}")
            print(f"  X_num tiene:   {list(X_num.columns)}")

            # Encontrar diferencias
            set_model = set(model_feat_names)
            set_x = set(X_num.columns)
            solo_model = set_model - set_x
            solo_x = set_x - set_model
            if solo_model:
                print(f"  Solo en modelo: {solo_model}")
            if solo_x:
                print(f"  Solo en X_num:  {solo_x}")
        else:
            print("  [OK] Orden de columnas COINCIDE.")

        # Intentar predecir
        print(f"\n  Intentando predict()...")
        try:
            y_pred = modelo.predict(X_num)
            print(f"  [OK] Predict exitoso! shape={y_pred.shape}, tipo={type(y_pred)}")
            print(f"  Predicciones (primeros 10): {np.round(y_pred[:10], 3)}")
        except Exception as e:
            print(f"  [ERROR] Predict FALLO: {type(e).__name__}: {e}")
            traceback.print_exc()

            # Intentar fallback con numpy
            print(f"\n  Intentando numpy fallback...")
            try:
                X_arr = X_num.values.astype(np.float64)
                y_pred = modelo.predict(X_arr)
                print(f"  [OK] Predict con numpy exitoso!")
            except Exception as e2:
                print(f"  [ERROR] Predict con numpy tambien FALLO: {type(e2).__name__}: {e2}")

print("\n" + "=" * 80)
print("DIAGNOSTICO COMPLETADO")
print("=" * 80)
