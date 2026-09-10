# AMR Sentinel

An AI-powered antimicrobial resistance (AMR) surveillance system — uses a pretrained protein language model (ESM2) to embed bacterial protein sequences, classifies resistance mechanisms against real CARD database gene families, and flags sequences that don't match known resistance patterns for surveillance follow-up.

## What this is

Most sequence-classification projects use hand-crafted features (k-mer frequencies, amino acid composition). This project instead uses **ESM2** — Meta's pretrained protein language model, trained on hundreds of millions of real protein sequences — to generate embeddings that capture functional/structural similarity, not just surface-level sequence composition. A classifier head trained on top of those embeddings predicts the resistance mechanism of a submitted protein sequence, and a separate novelty detector flags sequences that don't closely match any known resistance mechanism — the actual function a real AMR surveillance system needs, since emerging resistance mechanisms don't announce themselves; they show up as things that don't fit existing categories.

This project extends the antimicrobial-resistance focus of AMPROBE (a prior bio-entrepreneurship competition concept combining antimicrobial peptide design with CRISPR-based diagnostics) into a working, deployable AI system.

## Data source and licensing

Training data comes from the **Comprehensive Antibiotic Resistance Database (CARD)** — the authoritative, peer-reviewed reference database for antimicrobial resistance genes.

**CARD's data is not redistributed in this repository.** Per CARD's usage terms, reproduction of their data is restricted; this project's `.gitignore` excludes `data/raw/` entirely. To reproduce this project, download CARD's data yourself (free, no signup) from card.mcmaster.ca/download and place the files in `data/raw/` — see `data/prepare_dataset.py` for the exact expected filenames.

**Citation:** Alcock et al. 2023. *CARD 2023: expanded curation, support for machine learning, and resistance mechanism curation.* Nucleic Acids Research.

This project is built for personal/educational portfolio purposes, not redistribution or commercial use.

## How it works

1. A protein sequence is submitted via `/analyze`
2. **ESM2** (`esm2_t6_8M_UR50D` — the smallest checkpoint, CPU-only, no GPU required) generates a mean-pooled sequence embedding
3. A **RandomForest classifier**, trained on ESM2 embeddings of real CARD reference sequences, predicts the resistance mechanism with a confidence score
4. A **novelty detector** computes the sequence's cosine distance to the nearest known resistance-mechanism centroid in embedding space — sequences far from every known category are flagged `is_novel: true`
5. Every analysis is logged to Postgres for surveillance trend tracking over time

## Training data and honest performance reporting

6,058 labeled sequences were parsed from CARD, across 6 resistance mechanism classes. The data is **heavily imbalanced** (antibiotic inactivation — largely beta-lactamases — makes up ~87% of examples), which reflects real biology, not a data quality issue: enzymatic inactivation genuinely is the dominant resistance mechanism in nature.

Per-class performance (RandomForest, `class_weight="balanced"`, evaluated on a held-out 20% test split):

| Mechanism | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| antibiotic inactivation | 0.99 | 1.00 | 0.99 | 1049 |
| antibiotic target alteration | 0.96 | 0.83 | 0.89 | 58 |
| antibiotic efflux | 0.94 | 0.84 | 0.88 | 55 |
| antibiotic target protection | 0.94 | 1.00 | 0.97 | 34 |
| antibiotic target replacement | 1.00 | 0.93 | 0.96 | 14 |
| reduced permeability to antibiotic | 0.00 | 0.00 | 0.00 | 2 |

**Overall accuracy: 98%. Macro-average F1: 0.78** — the macro average is the more honest number, since it weights all classes equally regardless of size, rather than being dominated by the 87%-majority class.

The `reduced permeability to antibiotic` class scored 0 across the board — with only 2 test examples (8 total in the entire dataset), this is a sample-size floor, not a meaningful model failure; no classifier can be reliably evaluated on 2 examples. This is reported plainly rather than hidden, since honestly characterizing a model's limitations is part of doing this work correctly.

One additional simplification: CARD occasionally lists multiple resistance mechanisms for a single sequence (e.g., "antibiotic efflux;reduced permeability to antibiotic"). This project keeps only the first-listed mechanism per sequence for single-label classification — a real production surveillance system would ideally support multi-label prediction, which is a larger scope than this project targets.

## Core components

| Layer | What it does |
|---|---|
| `training/embed_sequences.py` | Runs CARD reference sequences through ESM2, caches embeddings to disk |
| `training/train_classifier.py` | Trains the RandomForest classifier + computes per-class centroids for novelty detection |
| `app/models/embedding_service.py` | Loads ESM2 once at startup, embeds incoming sequences at inference time |
| `app/models/classifier_service.py` | Wraps the trained classifier + label encoder |
| `app/models/novelty_detector.py` | Cosine-distance-to-nearest-centroid novelty scoring |
| `app/db/` | SQLAlchemy models + schema for surveillance logging |
| `app/api/` | `/analyze`, `/surveillance/summary`, `/health` |

## Tech stack

- Python 3.11, FastAPI, Uvicorn
- `fair-esm` + PyTorch (ESM2 protein language model)
- scikit-learn (classifier + novelty centroids)
- Biopython (FASTA parsing)
- PostgreSQL + SQLAlchemy
- pytest, pytest-mock
- Docker + docker-compose
- GitHub Actions CI

## Project structure

```text
amr-sentinel/
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── pytest.ini
├── data/
│ ├── raw/ # CARD downloads go here (gitignored)
│ └── prepare_dataset.py
├── training/
│ ├── embed_sequences.py
│ └── train_classifier.py
├── app/
│ ├── main.py
│ ├── config.py
│ ├── schemas.py
│ ├── db/
│ ├── models/
│ └── api/
└── tests/
├── test_models/
└── test_api/
```


## Running it locally

```bash
# 1. Download CARD data (see "Data source and licensing" above)
#    Place protein_fasta_protein_homolog_model.fasta and aro_index.tsv in data/raw/

# 2. Set up
cd amr-sentinel
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# 3. Parse the data, generate embeddings, train the classifier
python -m data.prepare_dataset
python -m training.embed_sequences     # slow — one ESM2 forward pass per sequence, CPU
python -m training.train_classifier

# 4. Run the unit test suite
python -m pytest -v

# 5. Bring up the full stack
docker-compose up --build

# 6. Smoke test
curl http://localhost:8001/health

curl -X POST http://localhost:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"sequence": "<a real protein sequence>"}'

curl "http://localhost:8001/surveillance/summary?window_minutes=60"
```

## Verified working example

A real CARD reference sequence (CblA-1, ARO:3002999 — a class A beta-lactamase) was submitted end-to-end through the full pipeline:

```json
{"predicted_mechanism":"antibiotic inactivation","confidence":0.988,"novelty_distance":0.0459,"is_novel":false,"sequence_length":296}
```

This is biologically correct: beta-lactamases resist antibiotics by enzymatically inactivating them, exactly matching the predicted mechanism, with high confidence and low novelty distance (as expected for a sequence the model was trained on the mechanism category of).

## License

MIT (project code). Training data is sourced from CARD under their own terms — see "Data source and licensing" above.
