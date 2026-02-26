import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd


DEFAULT_XLSX = (
    "Resultados_Modelo_stats_MegaPipeline/union_febrero_stats_para ppt.xlsx"
)
TRUTHY = {"true", "verdadero", "1", "si", "sí", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cuenta, por modelo, cuantas asignaturas tienen el menor MAE "
            "filtrando check usar = verdadero."
        )
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_XLSX,
        help=f"Ruta del archivo Excel (default: {DEFAULT_XLSX}).",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help=(
            "Hoja a leer. Si no se especifica, se detecta una hoja con columnas "
            "mean/metric/Modelo/check usar."
        ),
    )
    return parser.parse_args()


def to_bool_like(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in TRUTHY


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


def read_excel_with_lock_fallback(path: Path, sheet_name: str | None):
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except PermissionError:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / path.name
            shutil.copy2(path, tmp)
            return pd.read_excel(tmp, sheet_name=sheet_name)


def pick_sheet(path: Path) -> str:
    xl = pd.ExcelFile(path)
    required = {"mean", "metric", "modelo", "check usar"}
    for sheet in xl.sheet_names:
        head = pd.read_excel(path, sheet_name=sheet, nrows=1)
        cols = {str(c).strip().lower() for c in head.columns}
        if required.issubset(cols):
            return sheet
    raise ValueError(
        "No se encontro una hoja con columnas requeridas: mean, metric, Modelo, check usar."
    )


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: No existe archivo: {input_path}")
        return 1

    sheet = args.sheet
    if sheet is None:
        try:
            sheet = pick_sheet(input_path)
        except PermissionError:
            with tempfile.TemporaryDirectory() as td:
                tmp = Path(td) / input_path.name
                shutil.copy2(input_path, tmp)
                sheet = pick_sheet(tmp)

    df = read_excel_with_lock_fallback(input_path, sheet)

    required_columns = [
        "mean",
        "metric",
        "Modelo",
        "check usar",
        "Cod materia curso",
        "Descripcion_Materia",
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        print("ERROR: Faltan columnas requeridas:")
        for col in missing:
            print(f" - {col}")
        return 1

    work = df.copy()
    work["metric"] = work["metric"].astype(str).str.strip().str.upper()
    work["__is_mae"] = work["metric"].str.contains("MAE", na=False)
    work["__usar"] = work["check usar"].apply(to_bool_like)
    work["__mean_num"] = to_numeric_series(work["mean"])

    filtered = work[work["__is_mae"] & work["__usar"] & work["__mean_num"].notna()].copy()
    if filtered.empty:
        print("No hay filas MAE con check usar verdadero y mean numerico.")
        return 0

    subject_cols = ["Cod materia curso", "Descripcion_Materia"]
    min_mae = filtered.groupby(subject_cols, dropna=False, observed=False)["__mean_num"].min()
    best = filtered.merge(
        min_mae.rename("__min_mae"),
        left_on=subject_cols,
        right_index=True,
        how="inner",
    )
    best = best[best["__mean_num"] == best["__min_mae"]].copy()

    counts = (
        best.groupby("Modelo", dropna=False, observed=False)
        .size()
        .sort_values(ascending=False)
    )

    total_subjects = int(min_mae.shape[0])
    print(f"INPUT={input_path}")
    print(f"SHEET={sheet}")
    print(f"TOTAL_ASIGNATURAS_EVALUADAS={total_subjects}")
    for model, n in counts.items():
        print(f"{model}: {int(n)} asignaturas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
