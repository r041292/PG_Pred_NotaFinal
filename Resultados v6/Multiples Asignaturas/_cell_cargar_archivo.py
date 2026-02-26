# CARGAR ARCHIVO
print("=== Reivsar asignaturas. Carga Archivo Maestro.   ===\n")

#Flag para usar parquet o csv. El parquet fue creado del CSV original importando desde el otro metodo que genera el documento.
parquet_use= True

# Solicitar archivo de historial
while True:
    try:
        if parquet_use:
            ruta_historial = hitorial_parqet
            df_historial = pd.read_parquet(ruta_historial)
        else:
            ruta_historial = historial
            df_historial = pd.read_csv(ruta_historial, sep=';')
        print(f"✓ Archivo de historial cargado: {len(df_historial)} registros")
        df_historial_asignaturas_nombres = crear_df_asignaturas(df_historial)
        break
    except Exception as e:
        print(f"Error al cargar historial: {e}")
        print("Intente nuevamente.\n")