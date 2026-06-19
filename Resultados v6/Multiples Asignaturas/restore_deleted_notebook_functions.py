import json
import re
from pathlib import Path


NB_PATH = Path(
    r"C:\Users\Rubiel\OneDrive - Universidad del Norte\Maestria\Proyecto de grado\Prereq\Resultados v6\Multiples Asignaturas\rev_multiples_asig_rev_ia.ipynb"
)
UPDATE_SCRIPT = Path(
    r"C:\Users\Rubiel\OneDrive - Universidad del Norte\Maestria\Proyecto de grado\Prereq\Resultados v6\Multiples Asignaturas\update_rev_multi_notebook.py"
)


HELPERS_TO_RESTORE = """
def _tiene_modelo_guardado_por_task(asignatura, modelo, task, variant='main'):
    try:
        paths = get_model_paths(asignatura, modelo, variant, task)
        if os.path.isfile(paths['model']):
            return True
    except Exception:
        pass
    return _find_legacy_model_file(asignatura, modelo, task) is not None

def _tiene_par_modelos_guardados(asignatura, modelo, variant='main'):
    return (
        _tiene_modelo_guardado_por_task(asignatura, modelo, 'prediccion_retiro', variant=variant)
        and _tiene_modelo_guardado_por_task(asignatura, modelo, 'prediccion_nota', variant=variant)
    )

def elegir_asignaturas_desde_csv_o_historial(historial, cat_prereq=None, cant_asignaturas=5, csv_path=None, matricula_minima=200, omitir_si_modelos_existentes=True):
    # Obtiene asignaturas desde un CSV local si ya existe; si no, lo genera
    # a partir del historial con la misma logica base de conteo por asignatura.
    if csv_path is None:
        csv_path = NOTEBOOK_DIR / 'asignaturas_all.csv'
    else:
        csv_path = Path(csv_path)

    if cat_prereq is None or cat_prereq == 'all':
        df_historial = historial.copy()
    else:
        df_historial = historial[historial["Observacion_Prerrequisito"] == cat_prereq].copy()

    if csv_path.exists():
        print("[Resultados Funcion : INFO] ", f"[Asignaturas] Cargando asignaturas desde CSV existente: {csv_path}")
        df_filtrado = pd.read_csv(csv_path, sep=';', encoding='utf-8-sig')
    else:
        df_filtrado = (
            df_historial
            .groupby("Cod materia curso")
            .size()
            .reset_index(name="num_registros")
            .sort_values("num_registros", ascending=False)
            .reset_index(drop=True)
        )
        if matricula_minima is not None:
            df_filtrado = df_filtrado[df_filtrado["num_registros"] >= matricula_minima].reset_index(drop=True)

        df_filtrado.to_csv(csv_path, index=False, sep=';', encoding='utf-8-sig')
        print("[Resultados Funcion : INFO] ", f"[Asignaturas] CSV generado en: {csv_path}")

    if "num_registros" in df_filtrado.columns and matricula_minima is not None:
        df_filtrado = df_filtrado[df_filtrado["num_registros"] >= matricula_minima].reset_index(drop=True)

    lista_asignaturas_base = df_filtrado["Cod materia curso"].tolist()
    vista = lista_asignaturas_base[:10]
    sufijo = '...' if len(lista_asignaturas_base) > 10 else ''
    print("[Resultados Funcion : INFO] ", f"[Asignaturas] Lista base cargada ({len(lista_asignaturas_base)}): {vista}{sufijo}")

    if omitir_si_modelos_existentes:
        lista_asignaturas = []
        for asig in lista_asignaturas_base:
            modelos_completos = [
                modelo_nombre
                for modelo_nombre in ['xgboost', 'rf', 'catboost']
                if _tiene_par_modelos_guardados(asig, modelo_nombre, variant='main')
            ]
            if modelos_completos:
                print("[Resultados Funcion : INFO] ", f"[Asignaturas] Se omite {asig} porque ya tiene modelos completos: {modelos_completos}")
                continue
            lista_asignaturas.append(asig)
    else:
        print("[Resultados Funcion : INFO] ", "[Asignaturas] Validacion de modelos existentes desactivada por flag.")
        lista_asignaturas = lista_asignaturas_base

    lista_asignaturas = lista_asignaturas[:cant_asignaturas]
    print("[Resultados Funcion : INFO] ", f"[Asignaturas] Lista seleccionada ({len(lista_asignaturas)}): {lista_asignaturas}")
    return lista_asignaturas

def filtrar_asignaturas_por_matricula_minima(df_usar, asig_a_usar, matricula_minima=200):
    if matricula_minima is None:
        return df_usar, asig_a_usar

    conteo = (
        df_usar.groupby("Cod materia curso")
        .size()
        .reset_index(name="num_registros")
    )
    asignaturas_validas = conteo[conteo["num_registros"] >= matricula_minima]["Cod materia curso"].tolist()
    df_usar_filtrado = df_usar[df_usar["Cod materia curso"].isin(asignaturas_validas)].copy()
    lista_final = [a for a in asig_a_usar if a in set(asignaturas_validas)]

    print("[Resultados Funcion : INFO] ", f"[Asignaturas] Despues de filtro de matricula minima ({matricula_minima}), quedan {len(lista_final)} asignaturas.")
    return df_usar_filtrado, lista_final
"""


def extract_correr_modelos() -> str:
    text = UPDATE_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"def correr_modelos_multi_por_asignatura\(df_usar, variant_modelo_generar='main'\):.*?return df_resultados_final, df_resultados_stats, df_resultados_class_retiro_stats",
        text,
        re.S,
    )
    if not match:
        raise RuntimeError("No se pudo extraer correr_modelos_multi_por_asignatura desde update_rev_multi_notebook.py")

    lines = match.group(0).splitlines()
    normalized = [line[12:] if line.startswith("            ") else line for line in lines]
    return "\n".join(normalized)


def main():
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    cell5 = "".join(nb["cells"][5]["source"])
    if "def elegir_asignaturas_desde_csv_o_historial(" not in cell5:
        insert_at = cell5.index("## Crear DataFrame con los nombres de las asignaturas")
        cell5 = cell5[:insert_at] + HELPERS_TO_RESTORE + "\n\n" + cell5[insert_at:]

    cell15 = "".join(nb["cells"][15]["source"])
    if "def correr_modelos_multi_por_asignatura(" not in cell15:
        cell15 = extract_correr_modelos() + "\n\n" + cell15

    nb["cells"][5]["source"] = cell5.splitlines(keepends=True)
    nb["cells"][15]["source"] = cell15.splitlines(keepends=True)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("NOTEBOOK_FUNCTIONS_RESTORED")


if __name__ == "__main__":
    main()
