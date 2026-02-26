def correr_modelos_xgboost_por_asignatura(df_usar):
    """
    Ejecuta el pipeline completo de XGBoost (clasificación de retiro y regresión de nota)
    por asignatura, seleccionando automáticamente la lógica de preprocesamiento según
    la columna `tipo_asig_prereq`.

