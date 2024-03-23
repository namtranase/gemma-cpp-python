# gemma-cpp-python: Python Bindings for [gemma.cpp](https://github.com/google/gemma.cpp)

**Latest Version: v0.1.3.post3**
- Fixed absolute path for libsentencepiece.0.0.0.dylib
- Interface changes due to updates in gemma.cpp.
- Enhanced user experience for ease of use 🙏. Give it a try!

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`gemma-cpp-python` provides Python bindings for `gemma`, a high-performance C++ library, bridging advanced computational functionalities to Python.

## 🙏 Acknowledgments
Special thanks to the creators and contributors of [gemma.cpp](https://github.com/google/gemma.cpp) for their foundational work.

## 💬 Demo Chat and Chat with Website!
Check out the new chat demo included in the examples directory! This interactive interface showcases how you can engage in real-time conversations with the Gemma model.

For the Chat with Website, please visit the [tutorial](examples/webchat/README.md) for more detail/


### Using Gemma to chat with website
![Gemma Cpp Python Chat with Website Demo](asset/demo_chat_website.png)

### Chat with Gemma
![Gemma Cpp Python Chat Demo](asset/demo_chat.png)


## 🛠 Installation
`Prerequisites`: Ensure Python 3.8+ and pip are installed.

`System requirements`: For now, I only tested it on the Unix-like Platforms and the MacOS. Please visit the [gemma.cpp installation](https://github.com/google/gemma.cpp?tab=readme-ov-file#system-requirements) for more details.

`Models`: pygemma supported 2b-it-sfp model for now, to install model, [please visit here](https://github.com/google/gemma.cpp?tab=readme-ov-file#step-1-obtain-model-weights-and-tokenizer-from-kaggle-or-hugging-face-hub)

### Install from PyPI
For a quick setup, install directly from PyPI:
```bash
pip install pygemma==0.1.3
```

### For Developers: Install from Source
To install the latest version or for development purposes:

1. Clone the repo and enter the directory:
```bash
git clone https://github.com/namtranase/gemma-cpp-python.git
cd gemma-cpp-python
```

2. Install Python dependencies and pygemma:
```bash
pip install .
```

## 🖥 Usage

To acctually run the model, you need to install the model followed on the [gemma.cpp](https://github.com/google/gemma.cpp?tab=readme-ov-file#step-1-obtain-model-weights-and-tokenizer-from-kaggle) repo

For usage examples, refer to tests/test_chat.py. Here's a quick start:
```bash
from pygemma import Gemma
gemma = Gemma()
gemma.show_help()
gemma.show_config()
gemma.load_model("/path/to/tokenizer", "/path/to/compressed_weight/", "model_type")
gemma.completion("Write a poem")
```

To run the demo on your local machine:
```bash
cd gemma-cpp-python/examples
pip install -r requirements.txt
streamlit run streamlit_demo.py
```

## 🤝 Contributing
Contributions are welcome. Please clone the repository, push your changes to a new branch, and submit a pull request.

## License
gemma-cpp-python is MIT licensed. See the LICENSE file for details.
