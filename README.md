# gemma-cpp-python: Python Bindings for [gemma.cpp](https://github.com/google/gemma.cpp)

**Latest Version: v0.1.0**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

`gemma-cpp-python` provides Python bindings for `gemma`, a high-performance C++ library, bridging advanced computational functionalities to Python.

## 🙏 Acknowledgments
Special thanks to the creators and contributors of [gemma.cpp](https://github.com/google/gemma.cpp) for their foundational work.


## 🛠 Installation

```bash
git clone https://github.com/namtranase/gemma-cpp-python.git
git submodule update --init --recursive
pip install -r requirements.txt
pip install .
```

## 🖥 Usage

To acctually run the model, you need to install the model followed on the [gemma.cpp](https://github.com/google/gemma.cpp?tab=readme-ov-file#step-1-obtain-model-weights-and-tokenizer-from-kaggle) repo

For usage examples, refer to tests/test_chat.py. Here's a quick start:
```bash
import pygemma
pygemma.show_help()
```



## 🤝 Contributing
Contributions are welcome. Please clone the repository, push your changes to a new branch, and submit a pull request.

## License
gemma-cpp-python is MIT licensed. See the LICENSE file for details.
