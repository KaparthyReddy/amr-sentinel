"""
Runs every sequence in the parsed training table through ESM2 (Meta's
pretrained protein language model) to generate a fixed-size embedding
per sequence. Embeddings are cached to disk since this is the slowest
step in the pipeline (a transformer forward pass per sequence, on CPU)
and only needs to run once unless the training data changes.

Uses the smallest ESM2 checkpoint (esm2_t6_8M_UR50D, ~8M params) -
CPU-feasible, no GPU required. Larger ESM2 checkpoints exist and would
likely improve accuracy, but at a cost/time tradeoff not justified for
this project's scope.
"""

import os

import numpy as np
import pandas as pd
import torch
import esm

INPUT_PATH = "data/processed/training_sequences.csv"
OUTPUT_EMBEDDINGS_PATH = "data/processed/embeddings.npy"
OUTPUT_LABELS_PATH = "data/processed/labels.csv"

MODEL_NAME = "esm2_t6_8M_UR50D"
MAX_SEQUENCE_LENGTH = 1000  # truncate very long sequences to keep inference tractable


def load_model():
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def embed_sequence(model, batch_converter, sequence: str) -> np.ndarray:
    truncated = sequence[:MAX_SEQUENCE_LENGTH]
    data = [("seq", truncated)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        # Layer 6 is the final layer for the 6-layer esm2_t6 checkpoint
        results = model(tokens, repr_layers=[6], return_contacts=False)

    token_representations = results["representations"][6]
    # Mean-pool over sequence length (excluding BOS/EOS tokens) to get one
    # fixed-size vector per sequence, regardless of sequence length
    sequence_embedding = token_representations[0, 1:len(truncated) + 1].mean(0)
    return sequence_embedding.numpy()


def embed_all_sequences() -> None:
    df = pd.read_csv(INPUT_PATH)
    print(f"Embedding {len(df)} sequences with {MODEL_NAME}...")

    model, batch_converter = load_model()

    embeddings = []
    for i, sequence in enumerate(df["sequence"]):
        embeddings.append(embed_sequence(model, batch_converter, sequence))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)} done")

    embeddings_array = np.vstack(embeddings)
    os.makedirs("data/processed", exist_ok=True)
    np.save(OUTPUT_EMBEDDINGS_PATH, embeddings_array)

    df[["aro_accession", "resistance_mechanism"]].to_csv(OUTPUT_LABELS_PATH, index=False)

    print(f"Saved embeddings: {embeddings_array.shape} -> {OUTPUT_EMBEDDINGS_PATH}")
    print(f"Saved labels -> {OUTPUT_LABELS_PATH}")


if __name__ == "__main__":
    embed_all_sequences()
