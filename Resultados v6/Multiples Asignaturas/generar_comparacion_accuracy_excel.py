import argparse
from pathlib import Path
import re
import unicodedata

import pandas as pd


DEFAULT_XLSX = "Resultados_Modelo_stats_MegaPipeline/union_febrero_stats_para ppt.xlsx"
DEFAULT_SOURCE_SHEET = "union_febrero_stats"
DEFAULT_OUTPUT_SHEET = "comparacion_metrica"
TRUTHY = {"true", "verdadero", "1", "si", "sí", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera comparacion por asignatura para una metrica especifica "
            "filtrando check usar verdadero y la escribe en el mismo Excel."
        )
    )
    parser.add_argument("--input", default=DEFAULT_XLSX, help="Excel de entrada/salida.")
    parser.add_argument(
        "--write-input",
        default=None,
        help="Excel destino donde se escribe la hoja (default: mismo --input).",
    )
    parser.add_argument(
        "--source-sheet",
        default=DEFAULT_SOURCE_SHEET,
        help=f"Hoja origen (default: {DEFAULT_SOURCE_SHEET}).",
    )
    parser.add_argument(
        "--output-sheet",
        default=DEFAULT_OUTPUT_SHEET,
        help=f"Hoja destino (default: {DEFAULT_OUTPUT_SHEET}).",
    )
    parser.add_argument(
        "--metric",
        required=True,
        help="Metrica exacta a comparar (ej: 'Precision (CV)', 'Recall (CV)').",
    )
    return parser.parse_args()


def to_bool_like(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in TRUTHY


def to_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    has_comma = s.str.contains(",", regex=False)
    has_dot = s.str.contains(".", regex=False)
    both = has_comma & has_dot
    s = s.where(
        ~both, s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    )
    only_comma = has_comma & ~has_dot
    s = s.where(~only_comma, s.str.replace(",", ".", regex=False))
    s = s.str.replace(" ", "", regex=False)
    return pd.to_numeric(s, errors="coerce")


def normalize_metric_text(text: str) -> str:
    s = unicodedata.normalize("NFKD", str(text))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper().strip()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: No existe archivo: {input_path}")
        return 1
    write_path = Path(args.write_input) if args.write_input else input_path
    if not write_path.exists():
        print(f"ERROR: No existe archivo destino para escritura: {write_path}")
        return 1

    df = pd.read_excel(input_path, sheet_name=args.source_sheet)
    required = [
        "mean",
        "metric",
        "Modelo",
        "check usar",
        "Cod materia curso",
        "Descripcion_Materia",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: Faltan columnas requeridas:")
        for c in missing:
            print(f" - {c}")
        return 1

    w = df.copy()
    w["metric"] = w["metric"].astype(str).str.strip().str.upper()
    w["__metric_norm"] = w["metric"].apply(normalize_metric_text)
    w["__usar"] = w["check usar"].apply(to_bool_like)
    w["__mean_num"] = to_numeric_series(w["mean"])

    target_metric_norm = normalize_metric_text(args.metric)
    metric_mask = w["__metric_norm"].eq(target_metric_norm)

    filtered = w[metric_mask & w["__usar"] & w["__mean_num"].notna()].copy()

    subject_cols = ["Cod materia curso", "Descripcion_Materia"]
    if filtered.empty:
        summary = pd.DataFrame(
            {
                "Modelo": [],
                "Asignaturas_ganadas": [],
            }
        )
        detail = pd.DataFrame(
            {
                "Mensaje": [
                    f"No se encontraron filas de {args.metric} con check usar verdadero."
                ]
            }
        )
        total_subjects = 0
    else:
        best_by_subject = filtered.groupby(subject_cols, dropna=False, observed=False)[
            "__mean_num"
        ].max()
        best = filtered.merge(
            best_by_subject.rename("__best"),
            left_on=subject_cols,
            right_index=True,
            how="inner",
        )
        best = best[best["__mean_num"] == best["__best"]].copy()
        counts = (
            best.groupby("Modelo", dropna=False, observed=False)
            .size()
            .sort_values(ascending=False)
        )
        summary = counts.rename("Asignaturas_ganadas").reset_index()
        detail = best[
            subject_cols + ["Modelo", "metric", "__mean_num", "check usar"]
        ].copy()
        detail = detail.rename(columns={"__mean_num": "valor_accuracy"})
        total_subjects = int(best_by_subject.shape[0])

    meta = pd.DataFrame(
        {
            "clave": ["archivo", "hoja_origen", "hoja_salida", "total_asignaturas"],
            "valor": [
                str(input_path),
                args.source_sheet,
                args.output_sheet,
                total_subjects,
            ],
        }
    )
    meta = pd.concat(
        [
            meta,
            pd.DataFrame({"clave": ["metrica"], "valor": [args.metric]}),
        ],
        ignore_index=True,
    )

    try:
        with pd.ExcelWriter(
            write_path, mode="a", engine="openpyxl", if_sheet_exists="replace"
        ) as writer:
            meta.to_excel(writer, sheet_name=args.output_sheet, index=False, startrow=0)
            summary.to_excel(
                writer, sheet_name=args.output_sheet, index=False, startrow=6
            )
            detail.to_excel(
                writer, sheet_name=args.output_sheet, index=False, startrow=10
            )
    except PermissionError:
        print(
            "ERROR: No se pudo escribir en el Excel (archivo abierto). "
            "Cierralo y vuelve a ejecutar."
        )
        return 1

    print(f"INPUT={input_path}")
    print(f"WRITE_INPUT={write_path}")
    print(f"SOURCE_SHEET={args.source_sheet}")
    print(f"OUTPUT_SHEET={args.output_sheet}")
    print(f"METRIC={args.metric}")
    print(f"TOTAL_ASIGNATURAS_EVALUADAS={total_subjects}")
    if summary.empty:
        print("SIN_RESULTADOS_METRICA=1")
    else:
        for _, row in summary.iterrows():
            print(f"{row['Modelo']}: {int(row['Asignaturas_ganadas'])} asignaturas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
