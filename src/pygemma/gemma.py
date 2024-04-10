import multiprocessing
import os
from enum import Enum
import random
from typing import List, Optional

import _pygemma


class ModelTraining(Enum):
    GEMMA_IT = 0
    GEMMA_PT = 1


class ModelType(Enum):
    Gemma2B = 0
    Gemma7B = 1


class Gemma:
    def __init__(
        self,
        *,
        tokenizer_path: str,
        compressed_weights_path: str,
        model_type: ModelType,
        model_training: ModelTraining,
        n_threads: Optional[int] = None,
    ):
        self.tokenizer_path = tokenizer_path
        self.compressed_weights_path = compressed_weights_path
        self.model_type = model_type
        self.model_training = model_training

        self.n_threads = n_threads or max(multiprocessing.cpu_count() - 2, 1)

        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {self.tokenizer_path}")

        if not os.path.exists(self.compressed_weights_path):
            raise FileNotFoundError(
                f"Compressed weights not found: {self.compressed_weights_path}"
            )

        self.model = _pygemma.GemmaModel(
            self.tokenizer_path,
            self.compressed_weights_path,
            self.model_type.value,
            self.model_training.value,
            self.n_threads,
        )

        assert self.model

    @property
    def bos_token(self) -> int:
        assert self.model
        return self.model.bos_token

    @property
    def eos_token(self) -> int:
        assert self.model
        return self.model.eos_token

    def __call__(
        self,
        prompt: str,
        *,
        max_tokens: int = 2048,
        max_generated_tokens: int = 1024,
        temperature: float = 1.0,
        seed: Optional[int] = None,
        verbosity: int = 0,
    ) -> str:
        assert self.model

        seed = seed or random.randint(0, 2**32 - 1)

        return self.model.generate(
            prompt,
            max_tokens,
            max_generated_tokens,
            temperature,
            seed,
            verbosity,
        )

    def tokenize(
        self,
        text: str,
        add_bos: bool = True,
    ) -> List[int]:
        assert self.model
        return self.model.tokenize(text, add_bos)

    def detokenize(
        self,
        tokens: List[int],
    ) -> str:
        assert self.model
        return self.model.detokenize(tokens)
