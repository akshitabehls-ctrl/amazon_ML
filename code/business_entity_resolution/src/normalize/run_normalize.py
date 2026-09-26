"""Apply clean_name/clean_address to a dataframe, and a CLI to batch all 6 source files."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.common.path_utils import PROCESSED_ROOT, get_dataset_root
from src.common.schema import ADDRESS_FIELDS, NAME_FIELDS, RAW_COLUMNS
from src.normalize.address_cleaner import clean_address
from src.normalize.name_cleaner import clean_name

DATASET_ROOT = get_dataset_root()

FILES = {
    ("train", "source1"): DATASET_ROOT / "train" / "train_source1.tsv",
    ("train", "source2"): DATASET_ROOT / "train" / "train_source2.tsv",
    ("train", "source3"): DATASET_ROOT / "train" / "train_source3.tsv",
    ("test", "source1"): DATASET_ROOT / "test" / "test_source1.tsv",
    ("test", "source2"): DATASET_ROOT / "test" / "test_source2.tsv",
    ("test", "source3"): DATASET_ROOT / "test" / "test_source3.tsv",
}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply clean_name/clean_address to every row; return entity_id/country + name_*/addr_* columns."""
    names = df["business_name"].map(clean_name)
    countries = df["country"].fillna("")
    addresses = [
        clean_address(addr, country=country)
        for addr, country in zip(df["business_address"], countries)
    ]

    out = pd.DataFrame({"entity_id": df["entity_id"].values, "country": df["country"].values})
    for field in NAME_FIELDS:
        if field == "legal_suffix":
            out[f"name_{field}"] = [",".join(n[field]) for n in names]
        else:
            out[f"name_{field}"] = [n[field] for n in names]
    for field in ADDRESS_FIELDS:
        out[f"addr_{field}"] = [a[field] for a in addresses]
    return out


def _read_source_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(df.columns) == RAW_COLUMNS, f"{path}: unexpected columns {list(df.columns)}"
    return df


def run_all(out_dir: Path = PROCESSED_ROOT) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for (split, source), path in FILES.items():
        print(f"normalizing {split}/{source} from {path} ...")
        df = _read_source_tsv(path)
        cleaned = normalize_dataframe(df)
        out_path = out_dir / f"{split}_{source}.parquet"
        cleaned.to_parquet(out_path, index=False)
        print(f"  wrote {len(cleaned):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Normalize all 6 entity-resolution source files.")
    parser.add_argument("--out-dir", default=str(PROCESSED_ROOT))
    args = parser.parse_args()
    run_all(Path(args.out_dir))


if __name__ == "__main__":
    main()
