"""
Tests EmbeddingService's output shape/consistency without loading the
real ESM2 model (which is slow and would make this an integration test,
not a unit test). Mocks the underlying model and alphabet, verifying the
service correctly wires them together.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import torch


def test_embed_returns_correct_shape():
    with patch("app.models.embedding_service.esm.pretrained.load_model_and_alphabet") as mock_loader:
        mock_model = MagicMock()
        mock_alphabet = MagicMock()

        # Simulate a batch_converter call: returns (labels, strs, tokens)
        mock_alphabet.get_batch_converter.return_value = MagicMock(
            return_value=(["seq"], ["MKT"], torch.zeros((1, 5), dtype=torch.long))
        )

        # Simulate the model's forward pass output: a representation tensor
        # of shape (batch=1, seq_len=5, embedding_dim=320)
        mock_model.return_value = {
            "representations": {6: torch.rand(1, 5, 320)}
        }

        mock_loader.return_value = (mock_model, mock_alphabet)

        from app.models.embedding_service import EmbeddingService
        service = EmbeddingService(model_name="fake-model")

        embedding = service.embed("MKT")

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (320,)


def test_embed_truncates_long_sequences():
    with patch("app.models.embedding_service.esm.pretrained.load_model_and_alphabet") as mock_loader:
        mock_model = MagicMock()
        mock_alphabet = MagicMock()
        mock_batch_converter = MagicMock(
            return_value=(["seq"], ["X" * 1000], torch.zeros((1, 1002), dtype=torch.long))
        )
        mock_alphabet.get_batch_converter.return_value = mock_batch_converter
        mock_model.return_value = {"representations": {6: torch.rand(1, 1002, 320)}}
        mock_loader.return_value = (mock_model, mock_alphabet)

        from app.models.embedding_service import EmbeddingService
        service = EmbeddingService(model_name="fake-model")

        long_sequence = "M" * 5000  # far longer than MAX_SEQUENCE_LENGTH
        service.embed(long_sequence)

        # Confirm the batch converter was called with a truncated sequence,
        # not the full 5000-character input
        call_args = mock_batch_converter.call_args[0][0]
        submitted_sequence = call_args[0][1]
        assert len(submitted_sequence) == 1000
