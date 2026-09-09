"""
Parses CARD's (Comprehensive Antibiotic Resistance Database) downloaded
reference files into a clean training table: sequence + resistance
mechanism label.

CARD data is NOT bundled in this repo (see .gitignore) due to its usage
terms. Download it yourself from https://card.mcmaster.ca/download and
place these two files in data/raw/:
    - protein_fasta_protein_homolog_model.fasta
    - aro_index.tsv

Citation (include if you publish anything derived from this):
Alcock et al. 2023. CARD 2023: expanded curation, support for machine
learning, and resistance mechanism curation. Nucleic Acids Research.
"""

import os
import re

import pandas as pd
from Bio import SeqIO

RAW_DIR = "data/raw"
FASTA_PATH = os.path.join(RAW_DIR, "protein_fasta_protein_homolog_model.fasta")
ARO_INDEX_PATH = os.path.join(RAW_DIR, "aro_index.tsv")
OUTPUT_PATH = "data/processed/training_sequences.csv"


def _extract_aro_accession(fasta_header: str) -> str | None:
    """CARD FASTA headers embed an ARO accession like 'ARO:3002999' — this
    pulls that out so we can join against aro_index.tsv."""
    match = re.search(r"ARO:(\d+)", fasta_header)
    return f"ARO:{match.group(1)}" if match else None


MIN_CLASS_SIZE = 5


def _simplify_mechanism_label(mechanism: str) -> str:
    """CARD sometimes lists multiple resistance mechanisms separated by
    ';' for a single sequence. This keeps only the first-listed mechanism
    as a simplifying assumption for single-label classification — a real
    surveillance system would ideally support multi-label prediction, but
    that's a meaningfully larger scope than this project targets."""
    return mechanism.split(";")[0].strip()


def build_training_table() -> pd.DataFrame:
    if not os.path.exists(FASTA_PATH) or not os.path.exists(ARO_INDEX_PATH):
        raise FileNotFoundError(
            f"Expected CARD files not found in {RAW_DIR}/. "
            "Download them from https://card.mcmaster.ca/download and place them there."
        )

    aro_index = pd.read_csv(ARO_INDEX_PATH, sep="\t")
    aro_index.columns = [c.strip() for c in aro_index.columns]
    accession_col = next(c for c in aro_index.columns if "Accession" in c)
    mechanism_col = next(c for c in aro_index.columns if "Resistance Mechanism" in c)

    aro_lookup = dict(zip(aro_index[accession_col], aro_index[mechanism_col]))

    rows = []
    for record in SeqIO.parse(FASTA_PATH, "fasta"):
        accession = _extract_aro_accession(record.description)
        mechanism = aro_lookup.get(accession)

        if mechanism is None or not mechanism.strip():
            continue

        sequence = str(record.seq)
        if len(sequence) < 20:
            continue

        rows.append({
            "aro_accession": accession,
            "sequence": sequence,
            "resistance_mechanism": _simplify_mechanism_label(mechanism),
        })

    df = pd.DataFrame(rows)

    # Drop classes too small to learn or evaluate meaningfully
    class_counts = df["resistance_mechanism"].value_counts()
    valid_classes = class_counts[class_counts >= MIN_CLASS_SIZE].index
    dropped = class_counts[class_counts < MIN_CLASS_SIZE]
    if len(dropped) > 0:
        print(f"Dropping {len(dropped)} classes with < {MIN_CLASS_SIZE} examples: {dict(dropped)}")

    return df[df["resistance_mechanism"].isin(valid_classes)].reset_index(drop=True)

if __name__ == "__main__":
    df = build_training_table()
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Parsed {len(df)} labeled sequences")
    print("Resistance mechanism distribution:")
    print(df["resistance_mechanism"].value_counts())
