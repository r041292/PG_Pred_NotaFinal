import argparse
import csv
from pathlib import Path


DEFAULT_FOLDER = "Resultados_Modelo_Excel_MegaPipeline"
DEFAULT_OUTPUT = "union_febrero_resultados.csv"
DEFAULT_FILES = [
    "resultados_asig_prereq_mp_corr_cat2026-01-29235703.csv",
    "resultados_asig_prereq_mp_corr_cat2026-01-30134748.csv",
    "resultados_asig_prereq_mp_corr_cat2026-02-02185130.csv",
    "resultados_asig_prereq_mp_corr_cat2026-02-03031208.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Une multiples CSV en uno, conservando la union de todas las columnas "
            "(columnas faltantes en un archivo quedan vacias)."
        )
    )
    parser.add_argument(
        "--folder",
        default=DEFAULT_FOLDER,
        help=f"Carpeta donde estan los CSV (default: {DEFAULT_FOLDER}).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Nombre del archivo de salida (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="Delimitador CSV (default: ;).",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Codificacion de lectura para entrada (default: utf-8-sig).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=DEFAULT_FILES,
        help="Lista de archivos CSV a unir en orden.",
    )
    return parser.parse_args()


def collect_fieldnames(paths: list[Path], delimiter: str, encoding: str) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()

    for path in paths:
        with path.open("r", encoding=encoding, newline="") as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            header = next(reader, None)
            if not header:
                continue
            for col in header:
                if col not in seen:
                    seen.add(col)
                    fieldnames.append(col)
    return fieldnames


def merge_csvs(
    paths: list[Path],
    out_path: Path,
    fieldnames: list[str],
    delimiter: str,
    encoding: str,
) -> tuple[int, list[tuple[str, int]]]:
    per_file_counts: list[tuple[str, int]] = []
    total_rows = 0

    with out_path.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.DictWriter(
            out_f,
            fieldnames=fieldnames,
            delimiter=delimiter,
            extrasaction="ignore",
            restval="",
        )
        writer.writeheader()

        for path in paths:
            file_rows = 0
            with path.open("r", encoding=encoding, newline="") as in_f:
                reader = csv.DictReader(in_f, delimiter=delimiter)
                for row in reader:
                    writer.writerow(row)
                    file_rows += 1
            total_rows += file_rows
            per_file_counts.append((path.name, file_rows))

    return total_rows, per_file_counts


def main() -> int:
    args = parse_args()
    folder = Path(args.folder)
    input_paths = [folder / name for name in args.files]

    missing = [str(p) for p in input_paths if not p.exists()]
    if missing:
        print("ERROR: No se encontraron estos archivos:")
        for m in missing:
            print(f" - {m}")
        return 1

    out_path = folder / args.output
    fieldnames = collect_fieldnames(input_paths, args.delimiter, args.encoding)
    if not fieldnames:
        print("ERROR: No se detectaron encabezados en los CSV de entrada.")
        return 1

    total_rows, per_file_counts = merge_csvs(
        input_paths,
        out_path,
        fieldnames,
        args.delimiter,
        args.encoding,
    )

    for name, count in per_file_counts:
        print(f"FILE={name};ROWS={count}")
    print(f"OUT={out_path}")
    print(f"ROWS={total_rows}")
    print(f"COLS={len(fieldnames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
