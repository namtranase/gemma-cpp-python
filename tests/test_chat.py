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
    parser.add_argument(
        "--input", type=str, required=False, help="Input text to chat with the model. If None, Switch to Chat mode.",
        default="Hello."
    )
    # Now using the parsed arguments
    args = parser.parse_args()
    if args.input is not None:
        string = pygemma.completion(
            [
                "--tokenizer",
                args.tokenizer,
                "--compressed_weights",
                args.compressed_weights,
                "--model",
                args.model,
            ], args.input
        )
        print(string)
    else:
        return pygemma.chat_base(
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
    #  python tests/test_chat.py --tokenizer ../Model_Weight/tokenizer.spm --compressed_weights ../Model_Weight/2b-it-sfp.sbs --model 2b-it
