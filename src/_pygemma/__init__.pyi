from typing import List

class GemmaModel:
    def __init__(
        self,
        tokenizer_path: str,
        compressed_weights_path: str,
        model_type: int,
        model_training: int,
        n_threads: int,
    ) -> None:
        pass

    @property
    def bos_token(self) -> int: ...
    @property
    def eos_token(self) -> int: ...
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        max_generated_tokens: int,
        temperature: float,
        seed: int,
        verbosity: int,
    ) -> str: ...
    def tokenize(
        self,
        text: str,
        add_bos: bool = True,
    ) -> List[int]: ...
    def detokenize(
        self,
        tokens: List[int],
    ) -> str: ...
