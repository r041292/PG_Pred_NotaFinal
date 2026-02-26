import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "historia_todos_2019_202610.parquet"
DEFAULT_OUTPUT = "historia_todos_2019_202610_dpto_asig_poblado.parquet"
DEFAULT_PERIOD_COL = "Periodo"
DEFAULT_CODE_COL = "Cod materia curso"
DEFAULT_DPTO_COL = "_ Matricula detalle para analisis.DPTO Asignatura"
DEFAULT_TARGET_PERIOD = 202610


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pobla la columna de DPTO Asignatura en un periodo objetivo usando "
            "el ultimo valor historico (periodos anteriores) por Cod materia curso."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Parquet de entrada.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Parquet de salida.")
    parser.add_argument(
        "--period-col",
        default=DEFAULT_PERIOD_COL,
        help=f"Columna de periodo (default: {DEFAULT_PERIOD_COL}).",
    )
    parser.add_argument(
        "--code-col",
        default=DEFAULT_CODE_COL,
        help=f"Columna de codigo de materia (default: {DEFAULT_CODE_COL}).",
    )
    parser.add_argument(
        "--dpto-col",
        default=DEFAULT_DPTO_COL,
        help=f"Columna de departamento asignatura (default: {DEFAULT_DPTO_COL}).",
    )
    parser.add_argument(
        "--target-period",
        type=int,
        default=DEFAULT_TARGET_PERIOD,
        help=f"Periodo objetivo a poblar (default: {DEFAULT_TARGET_PERIOD}).",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Sobrescribe el archivo de entrada en lugar de escribir uno nuevo.",
    )
    return parser.parse_args()


def normalize_non_empty(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: No existe el archivo: {input_path}")
        return 1

    output_path = input_path if args.inplace else Path(args.output)

    df = pd.read_parquet(input_path)
    required = [args.period_col, args.code_col, args.dpto_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print("ERROR: Faltan columnas requeridas:")
        for c in missing:
            print(f" - {c}")
        return 1

    period_num = pd.to_numeric(df[args.period_col], errors="coerce")
    target_mask = period_num.eq(args.target_period)
    history_mask = period_num.lt(args.target_period)

    history = df.loc[history_mask, [args.code_col, args.period_col, args.dpto_col]].copy()
    history = history.loc[normalize_non_empty(history[args.dpto_col])].copy()
    history["__period_num"] = pd.to_numeric(history[args.period_col], errors="coerce")
    history = history.sort_values([args.code_col, "__period_num"])

    latest_dpto_by_code = history.groupby(
        args.code_col, dropna=False, observed=False
    )[args.dpto_col].last()

    mapped_values = df.loc[target_mask, args.code_col].map(latest_dpto_by_code)
    previous_values = df.loc[target_mask, args.dpto_col]
    new_values = mapped_values.where(mapped_values.notna(), previous_values)

    rows_target = int(target_mask.sum())
    rows_with_source = int(mapped_values.notna().sum())
    rows_changed = int((previous_values != new_values).fillna(False).sum())
    unique_codes_mapped = int(pd.Series(mapped_values.dropna().unique()).size)

    df.loc[target_mask, args.dpto_col] = new_values
    df.to_parquet(output_path, index=False)

    print(f"INPUT={input_path}")
    print(f"OUTPUT={output_path}")
    print(f"TARGET_PERIOD={args.target_period}")
    print(f"TARGET_ROWS={rows_target}")
    print(f"ROWS_WITH_HISTORY_VALUE={rows_with_source}")
    print(f"ROWS_CHANGED={rows_changed}")
    print(f"UNIQUE_CODES_MAPPED={unique_codes_mapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
