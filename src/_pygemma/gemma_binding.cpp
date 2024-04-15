#include "gemma_binding.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>

#include "gemma.h"  // Gemma
#include "util/app.h"
#include "util/args.h"

namespace py = pybind11;

GemmaModel::GemmaModel(const char *tokenizer_path_str,
                       const char *compressed_weights_path_str,
                       int model_type_id, int training_id, int num_threads) {
  const gcpp::Path tokenizer_path = gcpp::Path{tokenizer_path_str};
  const gcpp::Path compressed_weights_path =
      gcpp::Path{compressed_weights_path_str};
  const gcpp::Path weights_path = gcpp::Path{""};

  this->model_type = static_cast<gcpp::Model>(model_type_id);
  this->model_training = static_cast<gcpp::ModelTraining>(training_id);
  this->num_threads = static_cast<size_t>(num_threads);

  pool = new hwy::ThreadPool(num_threads);

  kv_cache = CreateKVCache(model_type);

  // For many-core, pinning threads to cores helps.
  if (this->num_threads > 10) {
    gcpp::PinThreadToCore(this->num_threads - 1);  // Main thread

    pool->Run(0, pool->NumThreads(), [](uint64_t /*task*/, size_t thread) {
      gcpp::PinThreadToCore(thread);
    });
  }

  model = new gcpp::Gemma(tokenizer_path, compressed_weights_path, weights_path,
                          model_type, model_training, *pool);
}

GemmaModel::~GemmaModel() { delete model; }

int GemmaModel::get_bos_token() const { return bos_token; }

int GemmaModel::get_eos_token() const { return eos_token; }

std::vector<int> GemmaModel::tokenize(const std::string &text,
                                      const bool add_bos) {
  std::vector<int> tokens;

  if (!model->Tokenizer()->Encode(text, &tokens).ok()) {
    throw std::runtime_error("Tokenization failed");
  }

  if (add_bos) {
    tokens.insert(tokens.begin(), bos_token);
  }

  return tokens;
}

std::string GemmaModel::detokenize(const std::vector<int> &tokens) {
  std::string text;
  if (!model->Tokenizer()->Decode(tokens, &text).ok()) {
    throw std::runtime_error("Detokenization failed");
  }
  return text;
}

std::string GemmaModel::generate(const std::string &prompt_string,
                                 size_t max_tokens, size_t max_generated_tokens,
                                 float temperature, uint_fast32_t seed,
                                 int verbosity) {
  size_t pos = 0;  // KV Cache position

  // Initialize random number generator
  std::mt19937 gen;
  gen.seed(seed);

  const std::string formatted = [&]() {
    if (model_training == gcpp::ModelTraining::GEMMA_IT) {
      return "<start_of_turn>user\n" + prompt_string +
             "<end_of_turn>\n<start_of_turn>model\n";
    }
    return prompt_string;
  }();

  std::vector<int> tokens = tokenize(formatted, true);
  size_t ntokens = tokens.size();

  std::string completion;

  // This callback function gets invoked everytime a token is generated
  const gcpp::StreamFunc stream_token = [this, &pos, &gen, &ntokens,
                                         tokenizer = model->Tokenizer(),
                                         &completion](int token, float) {
    ++pos;
    if (pos < ntokens) {
      // print feedback
    } else if (token != this->eos_token) {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      completion += token_text;
    }
    return true;
  };

  gcpp::GenerateGemma(*model,
                      {.max_tokens = max_tokens,
                       .max_generated_tokens = max_generated_tokens,
                       .temperature = temperature,
                       .verbosity = verbosity},
                      tokens, /*KV cache position = */ 0, kv_cache, *pool,
                      stream_token, gen);

  return completion;
}

PYBIND11_MODULE(_pygemma, m) {
  m.doc() = "Python binding for gemma.cpp";

  py::class_<GemmaModel>(m, "GemmaModel")
      .def_property_readonly("bos_token", &GemmaModel::get_bos_token,
                             "Get the BOS token")
      .def_property_readonly("eos_token", &GemmaModel::get_eos_token,
                             "Get the EOS token")
      .def(py::init<const char *, const char *, int, int, int>(),
           py::arg("tokenizer_path"), py::arg("compressed_weights_path"),
           py::arg("model_type"), py::arg("model_training"),
           py::arg("num_threads"), "Initialize the Gemma model")
      .def("tokenize", &GemmaModel::tokenize, py::arg("text"),
           py::arg("add_bos"),
           "Tokenize the input text and return the tokenized text")
      .def("detokenize", &GemmaModel::detokenize, py::arg("tokens"),
           "Detokenize the input tokens and return the detokenized text")
      .def("generate", &GemmaModel::generate, py::arg("prompt"),
           py::arg("max_tokens"), py::arg("max_generated_tokens"),
           py::arg("temperature"), py::arg("seed"), py::arg("verbosity"),
           "Generate text based on the input prompt");
}