import argparse
import pygemma


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

    args = parser.parse_args()

    # Now using the parsed arguments
    pygemma.chat_base(
        [
            "--tokenizer",
            args.tokenizer,
            "--compressed_weights",
            args.compressed_weights,
            "--model",
            args.model,
        ]
    )

    # Optionally, show help if needed
    # pygemma.show_help()


if __name__ == "__main__":
    main()
    # python tests/test_chat.py --tokenizer /path/to/tokenizer.spm --compressed_weights /path/to/weights.sbs --model model_identifier
