import argparse
import unicodedata
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path(
    r"Resultados_Modelo_v2/resultados_asig_xg_cat2026-02-23171813.csv"
)
DEFAULT_OUTPUT_SUBDIR = "propuestas_resultados_v2"
DEFAULT_OUTPUT_PREFIX = "propuesta_asignaturas_nrc"
DEFAULT_TOP_N = 100
DEFAULT_STD_FACTOR = 1.0
DEFAULT_SUBJECT_HIGH_STD_FACTOR = 1.0
DEFAULT_DELIMITER = ";"
REPITENCIA_STATES = {"Retiro", "Perdida"}
IGNORE_SUBJECTS = {
    "TALLER DE YOGA",
    "DESARROLLO PROFESIONAL",
    "EXAMEN DE LENGUA",
    "TALLER UNIV - PROY VIDA (MD)",
    "TENIS DE CAMPO",
    "LEADERSHIP AND TEAMWORK",
    "TALLER UNIV - PROY VIDA (NI)",
    "TALLER UNIV - PROY VIDA (DE)",
    "TIC INNOVACION Y DLLO SOCIEDAD",
    "TALLER UNIV - PROY VIDA (II)",
    "TALLER UNIV - PROY VIDA (AD)",
    "TALLER UNIV - PROY VIDA (PS)",
    "TALLER UNIV - PROY VIDA (IS)",
    "TALLER UNIV - PROY VIDA (IC)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construye una propuesta de cobertura por asignatura o NRC "
            "a partir de la prediccion final de XGBoost."
        )
    )
    parser.add_argument(
        "--input-csv",
        default=str(DEFAULT_INPUT),
        help=f"Ruta del CSV de entrada (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help=(
            "Directorio para los archivos de salida. Si no se indica, se usa "
            f"la carpeta hermana del CSV: <input>/../{DEFAULT_OUTPUT_SUBDIR}."
        ),
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help=f"Prefijo para los archivos de salida (default: {DEFAULT_OUTPUT_PREFIX}).",
    )
    parser.add_argument(
        "--delimiter",
        default=DEFAULT_DELIMITER,
        help=f"Delimitador del CSV (default: {DEFAULT_DELIMITER}).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"Cantidad de asignaturas a priorizar (default: {DEFAULT_TOP_N}).",
    )
    parser.add_argument(
        "--std-factor",
        type=float,
        default=DEFAULT_STD_FACTOR,
        help=(
            "Factor multiplicador de la desviacion estandar para marcar un NRC "
            "como sobresaliente dentro de su asignatura."
        ),
    )
    parser.add_argument(
        "--subject-high-std-factor",
        type=float,
        default=DEFAULT_SUBJECT_HIGH_STD_FACTOR,
        help=(
            "Factor multiplicador de la desviacion estandar para marcar una "
            "asignatura con repitencia general alta respecto al resto de "
            "asignaturas con repitencia."
        ),
    )
    return parser.parse_args()


def normalize_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    normalized = series.astype("string").str.strip()
    normalized = normalized.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    has_comma = normalized.str.contains(",", regex=False, na=False)
    has_dot = normalized.str.contains(".", regex=False, na=False)

    both = has_comma & has_dot
    comma_only = has_comma & ~has_dot

    normalized.loc[both] = (
        normalized.loc[both]
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    normalized.loc[comma_only] = normalized.loc[comma_only].str.replace(
        ",", ".", regex=False
    )

    return pd.to_numeric(normalized, errors="coerce")


def resolve_output_dir(input_csv: str | Path, output_dir: str | Path | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return Path(input_csv).parent / DEFAULT_OUTPUT_SUBDIR


def format_nrc_value(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return str(value)


def format_percent(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value) * 100:.1f}%"


def ascii_clean(value: object) -> str:
    text = "" if value is None else str(value)
    if "Ã" in text or "Â" in text:
        try:
            text = text.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def build_flow_visual(
    enriched_df: pd.DataFrame,
    subject_summary: pd.DataFrame,
    nrc_summary: pd.DataFrame,
    proposal: pd.DataFrame,
    top_n: int,
    std_factor: float,
    subject_high_std_factor: float,
) -> str:
    total_rows = len(enriched_df)
    ignored_rows = int(enriched_df["Asignatura a ignorar"].sum())
    base_rows = total_rows - ignored_rows
    total_subjects = int(subject_summary["Descripcion_Materia"].nunique())
    subjects_with_rep = int(subject_summary["repitencia_asignatura"].gt(0).sum())
    total_nrc = int(nrc_summary["nrc"].nunique())
    standout_nrc = int(nrc_summary["nrc_sobresaliente"].sum())

    selected = proposal.loc[proposal["seleccion_top_n"]]
    selected_subjects = len(selected)
    selected_nrc_only = int(selected["cobertura_recomendada"].eq("NRC").sum())
    selected_subject_only = int(selected["cobertura_recomendada"].eq("Asignatura").sum())
    selected_both = int(selected["cobertura_recomendada"].eq("Ambas").sum())
    high_subjects = int(subject_summary["asignatura_alta_repitencia"].sum())

    lines = [
        "=" * 96,
        "FLUJO DEL PROCESO - PROPUESTA DE COBERTURA",
        "=" * 96,
        "",
        "[CSV DE ENTRADA]",
        f"  {ascii_clean(DEFAULT_INPUT)}",
        f"  filas={total_rows}",
        "",
        "        |",
        "        v",
        "[1] NORMALIZAR Prediccion_final_XGB",
        "  - convertir coma decimal a punto",
        "  - crear Prediccion_final_XGB_num",
        "",
        "        |",
        "        v",
        "[2] CLASIFICAR Estado XGBOOST",
        "  - Retiro si prediccion < 0",
        "  - Perdida si 0 <= prediccion < 3",
        "  - Aprobacion si prediccion >= 3",
        "",
        "        |",
        "        v",
        "[3] MARCAR ASIGNATURAS A IGNORAR",
        f"  filas_ignoradas={ignored_rows}",
        f"  filas_utiles={base_rows}",
        "",
        "        |",
        "        v",
        "[4] CALCULAR REPITENCIA",
        f"  asignaturas_utiles={total_subjects}",
        f"  asignaturas_con_repitencia={subjects_with_rep}",
        f"  nrc_unicos={total_nrc}",
        "",
        "        |",
        "        +-------------------------------+",
        "        v                               v",
        "[5A] REPITENCIA ALTA DE ASIGNATURA     [5B] NRC SOBRESALIENTE",
        f"  umbral = media + {subject_high_std_factor:g} * desv",
        f"  umbral = media + {std_factor:g} * desv dentro de la asignatura",
        f"  asignaturas_altas={high_subjects}",
        f"  nrc_sobresalientes={standout_nrc}",
        "",
        "        \\                               /",
        "         \\                             /",
        "          v                           v",
        "[6] INTEGRAR REGLAS Y PRIORIZAR TOP N",
        f"  top_n={top_n}",
        f"  asignaturas_seleccionadas={selected_subjects}",
        "",
        "        |",
        "        v",
        "[7] GRUPOS FINALES",
        f"  - Apoyo en NRC: {selected_nrc_only}",
        f"  - Apoyo en Asignatura: {selected_subject_only}",
        f"  - Apoyo en Ambas: {selected_both}",
        "",
        "        |",
        "        v",
        "[8] SALIDAS",
        "  - CSV enriquecido",
        "  - Resumen por asignatura",
        "  - Detalle por NRC",
        "  - Propuesta top",
        "  - Propuesta de acciones",
        "  - Visual ASCII por grupos",
        "  - Flujo ASCII del proceso",
    ]

    return "\n".join(lines).rstrip() + "\n"


def build_text_visual(proposal: pd.DataFrame, top_n: int) -> str:
    selected = proposal.loc[proposal["seleccion_top_n"]].copy()
    total_selected = len(selected)
    groups = [
        ("APOYO EN NRC", "NRC"),
        ("APOYO EN ASIGNATURA", "Asignatura"),
        ("APOYO EN AMBAS", "Ambas"),
    ]

    lines = [
        "=" * 96,
        "PROPUESTA DE COBERTURA - REPRESENTACION TEXTUAL",
        "=" * 96,
        (
            f"Top N seleccionado: {top_n} | Asignaturas seleccionadas: {total_selected} | "
            f"NRC: {int(selected['cobertura_recomendada'].eq('NRC').sum())} | "
            f"Asignatura: {int(selected['cobertura_recomendada'].eq('Asignatura').sum())} | "
            f"Ambas: {int(selected['cobertura_recomendada'].eq('Ambas').sum())}"
        ),
        "",
    ]

    for idx, (title, group_name) in enumerate(groups, start=1):
        group_df = selected.loc[
            selected["cobertura_recomendada"].eq(group_name)
        ].sort_values("rank_prioridad")

        lines.append("-" * 96)
        lines.append(f"[{idx}] {title} ({len(group_df)})")
        lines.append("-" * 96)

        if group_df.empty:
            lines.append("  (sin asignaturas en este grupo)")
            lines.append("")
            continue

        for row in group_df.itertuples(index=False):
            subject_line = (
                f"|-- #{row.rank_prioridad} {ascii_clean(row.Descripcion_Materia)} "
                f"[rep_asig={int(row.repitencia_asignatura)}, "
                f"tasa_asig={format_percent(row.tasa_repitencia_asignatura)}]"
            )
            lines.append(subject_line)

            if row.cobertura_recomendada == "Asignatura":
                lines.append(
                    f"|   `-- Asignatura: {ascii_clean(row.criterio_cobertura_asignatura)}"
                )
            elif row.cobertura_recomendada == "NRC":
                lines.append(
                    "|   `-- NRC "
                    f"{format_nrc_value(row.nrc_priorizado)} "
                    f"[rep_nrc={int(row.repitencia_nrc_priorizado)}, "
                    f"tasa_nrc={format_percent(row.tasa_repitencia_nrc_priorizado)}]"
                )
            else:
                lines.append(
                    f"|   |-- Asignatura: {ascii_clean(row.criterio_cobertura_asignatura)}"
                )
                lines.append(
                    "|   `-- NRC "
                    f"{format_nrc_value(row.nrc_priorizado)} "
                    f"[rep_nrc={int(row.repitencia_nrc_priorizado)}, "
                    f"tasa_nrc={format_percent(row.tasa_repitencia_nrc_priorizado)}]"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def next_available_path(path: Path, suffix_base: str = "_updated") -> Path:
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix_base}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def write_dataframe_with_fallback(df: pd.DataFrame, path: Path) -> Path:
    try:
        df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")
        return path
    except PermissionError:
        fallback = next_available_path(path)
        df.to_csv(fallback, sep=";", index=False, encoding="utf-8-sig")
        return fallback


def write_text_with_fallback(text: str, path: Path) -> Path:
    try:
        path.write_text(text, encoding="utf-8")
        return path
    except PermissionError:
        fallback = next_available_path(path)
        fallback.write_text(text, encoding="utf-8")
        return fallback


def load_input(input_csv: Path, delimiter: str) -> pd.DataFrame:
    return pd.read_csv(input_csv, sep=delimiter, low_memory=False)


def build_enriched_dataset(df: pd.DataFrame) -> pd.DataFrame:
    enriched_df = df.copy()
    pred = normalize_numeric_series(enriched_df["Prediccion_final_XGB"])

    estado = pd.Series(pd.NA, index=enriched_df.index, dtype="object")
    estado.loc[pred < 0] = "Retiro"
    estado.loc[(pred >= 0) & (pred < 3)] = "Perdida"
    estado.loc[pred >= 3] = "Aprobacion"

    enriched_df["Prediccion_final_XGB_num"] = pred
    enriched_df["Estado XGBOOST"] = estado
    enriched_df["Asignatura a ignorar"] = enriched_df["Descripcion_Materia"].isin(
        IGNORE_SUBJECTS
    )
    enriched_df["Es_repitencia"] = (
        enriched_df["Estado XGBOOST"].isin(REPITENCIA_STATES).astype("int64")
    )
    return enriched_df


def summarize_subjects(base_df: pd.DataFrame) -> pd.DataFrame:
    subject_summary = (
        base_df.groupby("Descripcion_Materia", dropna=False)
        .agg(
            registros_asignatura=("Descripcion_Materia", "size"),
            repitencia_asignatura=("Es_repitencia", "sum"),
        )
        .reset_index()
    )
    subject_summary["tasa_repitencia_asignatura"] = (
        subject_summary["repitencia_asignatura"]
        / subject_summary["registros_asignatura"]
    )
    return subject_summary


def enrich_subjects_with_global_threshold(
    subject_summary: pd.DataFrame, subject_high_std_factor: float
) -> pd.DataFrame:
    positive_rep = subject_summary.loc[
        subject_summary["repitencia_asignatura"] > 0, "repitencia_asignatura"
    ]
    mean_rep = float(positive_rep.mean()) if not positive_rep.empty else 0.0
    std_rep = float(positive_rep.std(ddof=0)) if not positive_rep.empty else 0.0
    threshold = mean_rep + subject_high_std_factor * std_rep

    enriched = subject_summary.copy()
    enriched["media_global_repitencia_asignatura"] = mean_rep
    enriched["desv_global_repitencia_asignatura"] = std_rep
    enriched["umbral_alta_repitencia_asignatura"] = threshold
    enriched["asignatura_alta_repitencia"] = (
        enriched["repitencia_asignatura"] > threshold
    )
    return enriched


def summarize_nrcs(base_df: pd.DataFrame) -> pd.DataFrame:
    nrc_summary = (
        base_df.groupby(["Descripcion_Materia", "nrc"], dropna=False)
        .agg(
            registros_nrc=("nrc", "size"),
            repitencia_nrc=("Es_repitencia", "sum"),
        )
        .reset_index()
    )
    nrc_summary["tasa_repitencia_nrc"] = (
        nrc_summary["repitencia_nrc"] / nrc_summary["registros_nrc"]
    )
    return nrc_summary


def compute_nrc_stats(
    nrc_summary: pd.DataFrame, std_factor: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nrc_stats = (
        nrc_summary.groupby("Descripcion_Materia", dropna=False)
        .agg(
            total_nrc=("nrc", "nunique"),
            media_repitencia_nrc=("repitencia_nrc", "mean"),
            desv_repitencia_nrc=("repitencia_nrc", lambda s: s.std(ddof=0)),
            max_repitencia_nrc=("repitencia_nrc", "max"),
            media_tasa_nrc=("tasa_repitencia_nrc", "mean"),
            desv_tasa_nrc=("tasa_repitencia_nrc", lambda s: s.std(ddof=0)),
            max_tasa_nrc=("tasa_repitencia_nrc", "max"),
        )
        .reset_index()
    )

    nrc_stats["desv_repitencia_nrc"] = nrc_stats["desv_repitencia_nrc"].fillna(0.0)
    nrc_stats["desv_tasa_nrc"] = nrc_stats["desv_tasa_nrc"].fillna(0.0)

    nrc_detail = nrc_summary.merge(
        nrc_stats,
        on="Descripcion_Materia",
        how="left",
    )
    nrc_detail["umbral_repitencia_nrc"] = (
        nrc_detail["media_repitencia_nrc"]
        + std_factor * nrc_detail["desv_repitencia_nrc"]
    )
    nrc_detail["nrc_sobresaliente"] = (
        (nrc_detail["repitencia_nrc"] > 0)
        & (nrc_detail["repitencia_nrc"] > nrc_detail["umbral_repitencia_nrc"])
    )

    top_nrc = (
        nrc_detail.loc[nrc_detail["nrc_sobresaliente"]]
        .sort_values(
            [
                "Descripcion_Materia",
                "repitencia_nrc",
                "tasa_repitencia_nrc",
                "registros_nrc",
                "nrc",
            ],
            ascending=[True, False, False, False, True],
        )
        .drop_duplicates("Descripcion_Materia")
        .rename(
            columns={
                "nrc": "nrc_priorizado",
                "registros_nrc": "registros_nrc_priorizado",
                "repitencia_nrc": "repitencia_nrc_priorizado",
                "tasa_repitencia_nrc": "tasa_repitencia_nrc_priorizado",
                "umbral_repitencia_nrc": "umbral_repitencia_nrc_priorizado",
            }
        )[
            [
                "Descripcion_Materia",
                "nrc_priorizado",
                "registros_nrc_priorizado",
                "repitencia_nrc_priorizado",
                "tasa_repitencia_nrc_priorizado",
                "umbral_repitencia_nrc_priorizado",
            ]
        ]
    )

    return nrc_detail, top_nrc


def build_proposal(
    input_csv: str | Path = DEFAULT_INPUT,
    delimiter: str = DEFAULT_DELIMITER,
    top_n: int = DEFAULT_TOP_N,
    std_factor: float = DEFAULT_STD_FACTOR,
    subject_high_std_factor: float = DEFAULT_SUBJECT_HIGH_STD_FACTOR,
) -> dict[str, object]:
    input_path = Path(input_csv)
    raw_df = load_input(input_path, delimiter=delimiter)
    enriched_df = build_enriched_dataset(raw_df)

    base_df = enriched_df.loc[~enriched_df["Asignatura a ignorar"]].copy()
    subject_summary = enrich_subjects_with_global_threshold(
        summarize_subjects(base_df),
        subject_high_std_factor=subject_high_std_factor,
    )
    nrc_summary = summarize_nrcs(base_df)
    nrc_detail, top_nrc = compute_nrc_stats(nrc_summary, std_factor=std_factor)

    proposal = subject_summary.merge(top_nrc, on="Descripcion_Materia", how="left")
    proposal["tiene_nrc_priorizado"] = proposal["nrc_priorizado"].notna()

    proposal = proposal.sort_values(
        [
            "repitencia_asignatura",
            "tasa_repitencia_asignatura",
            "repitencia_nrc_priorizado",
            "tasa_repitencia_nrc_priorizado",
            "Descripcion_Materia",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    proposal["rank_prioridad"] = proposal.index + 1
    proposal["seleccion_top_n"] = (
        proposal["repitencia_asignatura"].gt(0)
        & proposal["rank_prioridad"].le(top_n)
    )
    proposal["cubrir_nrc"] = proposal["seleccion_top_n"] & proposal["tiene_nrc_priorizado"]
    proposal["cubrir_asignatura"] = proposal["seleccion_top_n"] & (
        ~proposal["tiene_nrc_priorizado"] | proposal["asignatura_alta_repitencia"]
    )
    proposal["cobertura_recomendada"] = "Sin cobertura"
    proposal.loc[
        proposal["cubrir_asignatura"] & ~proposal["cubrir_nrc"],
        "cobertura_recomendada",
    ] = "Asignatura"
    proposal.loc[
        ~proposal["cubrir_asignatura"] & proposal["cubrir_nrc"],
        "cobertura_recomendada",
    ] = "NRC"
    proposal.loc[
        proposal["cubrir_asignatura"] & proposal["cubrir_nrc"],
        "cobertura_recomendada",
    ] = "Ambas"
    proposal["criterio_cobertura_asignatura"] = pd.NA
    proposal.loc[
        proposal["cubrir_asignatura"] & proposal["asignatura_alta_repitencia"],
        "criterio_cobertura_asignatura",
    ] = (
        "Asignatura con repitencia general superior a media + "
        f"{subject_high_std_factor:g} desviacion estandar"
    )
    proposal.loc[
        proposal["cubrir_asignatura"] & ~proposal["asignatura_alta_repitencia"],
        "criterio_cobertura_asignatura",
    ] = "Cobertura general de la asignatura sin NRC sobresaliente"
    proposal["criterio_cobertura_nrc"] = pd.NA
    proposal.loc[proposal["cubrir_nrc"], "criterio_cobertura_nrc"] = (
        "NRC con repitencia superior a media + "
        f"{std_factor:g} desviacion estandar de la asignatura"
    )

    action_rows: list[dict[str, object]] = []
    selected = proposal.loc[proposal["seleccion_top_n"]].copy()
    for row in selected.itertuples(index=False):
        if row.cubrir_asignatura:
            action_rows.append(
                {
                    "rank_prioridad": row.rank_prioridad,
                    "Descripcion_Materia": row.Descripcion_Materia,
                    "nivel_accion": "Asignatura",
                    "elemento_priorizado": row.Descripcion_Materia,
                    "repitencia_asignatura": row.repitencia_asignatura,
                    "tasa_repitencia_asignatura": row.tasa_repitencia_asignatura,
                    "nrc_priorizado": pd.NA,
                    "repitencia_nrc_priorizado": pd.NA,
                    "tasa_repitencia_nrc_priorizado": pd.NA,
                    "criterio_cobertura": row.criterio_cobertura_asignatura,
                    "cobertura_recomendada_asignatura": row.cobertura_recomendada,
                }
            )
        if row.cubrir_nrc:
            action_rows.append(
                {
                    "rank_prioridad": row.rank_prioridad,
                    "Descripcion_Materia": row.Descripcion_Materia,
                    "nivel_accion": "NRC",
                    "elemento_priorizado": (
                        f"{row.Descripcion_Materia} | NRC {format_nrc_value(row.nrc_priorizado)}"
                    ),
                    "repitencia_asignatura": row.repitencia_asignatura,
                    "tasa_repitencia_asignatura": row.tasa_repitencia_asignatura,
                    "nrc_priorizado": row.nrc_priorizado,
                    "repitencia_nrc_priorizado": row.repitencia_nrc_priorizado,
                    "tasa_repitencia_nrc_priorizado": row.tasa_repitencia_nrc_priorizado,
                    "criterio_cobertura": row.criterio_cobertura_nrc,
                    "cobertura_recomendada_asignatura": row.cobertura_recomendada,
                }
            )

    proposal_actions = pd.DataFrame(action_rows)
    if not proposal_actions.empty:
        proposal_actions = proposal_actions.sort_values(
            [
                "rank_prioridad",
                "nivel_accion",
                "repitencia_nrc_priorizado",
                "Descripcion_Materia",
            ],
            ascending=[True, True, False, True],
            na_position="last",
        ).reset_index(drop=True)

    flow_visual = build_flow_visual(
        enriched_df=enriched_df,
        subject_summary=subject_summary,
        nrc_summary=nrc_detail,
        proposal=proposal,
        top_n=top_n,
        std_factor=std_factor,
        subject_high_std_factor=subject_high_std_factor,
    )
    text_visual = build_text_visual(proposal, top_n=top_n)

    return {
        "enriched_df": enriched_df,
        "subject_summary": subject_summary.sort_values(
            ["repitencia_asignatura", "tasa_repitencia_asignatura", "Descripcion_Materia"],
            ascending=[False, False, True],
        ).reset_index(drop=True),
        "nrc_summary": nrc_detail.sort_values(
            ["repitencia_nrc", "tasa_repitencia_nrc", "Descripcion_Materia", "nrc"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True),
        "proposal": proposal,
        "proposal_actions": proposal_actions,
        "flow_visual": flow_visual,
        "text_visual": text_visual,
    }


def save_outputs(
    outputs: dict[str, object],
    output_dir: str | Path,
    output_prefix: str = DEFAULT_OUTPUT_PREFIX,
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "enriched_df": output_path / f"{output_prefix}_enriquecido.csv",
        "subject_summary": output_path / f"{output_prefix}_resumen_asignaturas.csv",
        "nrc_summary": output_path / f"{output_prefix}_detalle_nrc.csv",
        "proposal": output_path / f"{output_prefix}_propuesta_top.csv",
        "proposal_actions": output_path / f"{output_prefix}_propuesta_acciones.csv",
        "flow_visual": output_path / f"{output_prefix}_flujo.txt",
        "text_visual": output_path / f"{output_prefix}_visual.txt",
    }

    dataframe_keys = [
        "enriched_df",
        "subject_summary",
        "nrc_summary",
        "proposal",
        "proposal_actions",
    ]
    for key in dataframe_keys:
        paths[key] = write_dataframe_with_fallback(outputs[key], paths[key])

    paths["flow_visual"] = write_text_with_fallback(
        str(outputs["flow_visual"]),
        paths["flow_visual"],
    )
    paths["text_visual"] = write_text_with_fallback(
        str(outputs["text_visual"]),
        paths["text_visual"],
    )

    return paths


def main() -> int:
    args = parse_args()
    output_dir = resolve_output_dir(args.input_csv, args.output_dir)
    outputs = build_proposal(
        input_csv=args.input_csv,
        delimiter=args.delimiter,
        top_n=args.top_n,
        std_factor=args.std_factor,
        subject_high_std_factor=args.subject_high_std_factor,
    )
    saved_paths = save_outputs(
        outputs,
        output_dir=output_dir,
        output_prefix=args.output_prefix,
    )

    proposal = outputs["proposal"]
    selected_count = int(proposal["seleccion_top_n"].sum())
    selected_actions = outputs["proposal_actions"]
    nrc_count = int(
        proposal.loc[proposal["seleccion_top_n"], "cubrir_nrc"].sum()
    )
    subject_count = int(
        proposal.loc[proposal["seleccion_top_n"], "cubrir_asignatura"].sum()
    )
    both_count = int(
        proposal.loc[proposal["seleccion_top_n"], "cobertura_recomendada"].eq("Ambas").sum()
    )

    print(f"INPUT={Path(args.input_csv)}")
    print(f"OUTPUT_DIR={output_dir}")
    print(f"TOP_N={args.top_n}")
    print(f"STD_FACTOR={args.std_factor}")
    print(f"SUBJECT_HIGH_STD_FACTOR={args.subject_high_std_factor}")
    print(f"ASIGNATURAS_CON_REPITENCIA={int(proposal['repitencia_asignatura'].gt(0).sum())}")
    print(f"SELECCIONADAS={selected_count}")
    print(f"ACCIONES_TOTALES={len(selected_actions)}")
    print(f"SELECCION_NRC={nrc_count}")
    print(f"SELECCION_ASIGNATURA={subject_count}")
    print(f"SELECCION_AMBAS={both_count}")
    for key, path in saved_paths.items():
        print(f"OUT_{key.upper()}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
