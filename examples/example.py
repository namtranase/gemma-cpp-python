from pygemma import Gemma, ModelType, ModelTraining
from time import time

TOKENIZER_PATH = "../model/tokenizer.spm"
COMPRESSED_WEIGHTS_PATH = "../model/2b-it-mqa.sbs"
MODEL_TYPE = ModelType.Gemma2B
MODEL_TRAINING = ModelTraining.GEMMA_IT


def main():
    gemma = Gemma(
        tokenizer_path=TOKENIZER_PATH,
        compressed_weights_path=COMPRESSED_WEIGHTS_PATH,
        model_type=MODEL_TYPE,
        model_training=MODEL_TRAINING,
    )

    start_time = time()
    res = gemma("Hello world!")
    print(f"Generated: {res}")

    print(f"Elapsed time: {time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
