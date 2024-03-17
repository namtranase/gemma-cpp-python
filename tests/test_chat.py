import argparse
from pygemma import Gemma


def main():
    parser = argparse.ArgumentParser(
        description="Test script for pygemma Python bindings."
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to the tokenizer file."
    )
    parser.add_argument(
        "--compressed_weights",
        type=str,
        required=True,
        help="Path to the compressed weights file.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model type identifier."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Input text to chat with the model. If None, Switch to Chat mode.",
        default="Hello.",
    )
    # Now using the parsed arguments
    args = parser.parse_args()

    gemma = Gemma()
    gemma.show_config()
    gemma.show_help()
    gemma.load_model(args.tokenizer, args.compressed_weights, args.model)
    gemma.completion("Write a poem")
    gemma.completion("What is the best war in history")


if __name__ == "__main__":
    main()
    #  python tests/test_chat.py --tokenizer ../Model_Weight/tokenizer.spm --compressed_weights ../Model_Weight/2b-it-sfp.sbs --model 2b-it
