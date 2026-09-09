"""
Loads ESM2 once at startup and generates embeddings for incoming
sequences at inference time - same embedding procedure used during
training (mean-pooled final-layer token representations), which matters:
a mismatch here would silently make the trained classifier and centroids
meaningless, since they'd be comparing embeddings computed two different
ways.
"""

import torch
import esm

from app.config import settings

MAX_SEQUENCE_LENGTH = 1000


class EmbeddingService:

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.esm_model_name
        self.model, alphabet = esm.pretrained.load_model_and_alphabet(self.model_name)
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.repr_layer = 6  # final layer for esm2_t6_* checkpoints

    def embed(self, sequence: str):
        truncated = sequence[:MAX_SEQUENCE_LENGTH]
        data = [("seq", truncated)]
        _, _, tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(tokens, repr_layers=[self.repr_layer], return_contacts=False)

        token_representations = results["representations"][self.repr_layer]
        sequence_embedding = token_representations[0, 1:len(truncated) + 1].mean(0)
        return sequence_embedding.numpy()
