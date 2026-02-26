def correr_modelos_multi_por_asignatura(df_usar):
    """
    Ejecuta pipeline completo por asignatura con XGBoost, RandomForest y CatBoost
    (clasificacion de retiro y regresion de nota) manteniendo la misma logica
    de seleccion de asignaturas, presentacion de resultados y determinacion
    de variables.

