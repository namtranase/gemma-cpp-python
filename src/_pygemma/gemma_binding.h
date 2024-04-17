#include "gemma.h"  // Gemma

#pragma once
class GemmaModel {
 private:
  gcpp::Gemma *model;
  gcpp::Model model_type;
  gcpp::ModelTraining model_training;

  size_t num_threads;
  hwy::ThreadPool *pool;
  gcpp::KVCache kv_cache;

  const int eos_token = 1;
  const int bos_token = 2;

 public:
  GemmaModel(const char *tokenizer_path_str,
             const char *compressed_weights_path_str, int model_type_id,
             int training_id, int num_threads);

  ~GemmaModel();

  int get_bos_token() const;

  int get_eos_token() const;

  std::vector<int> tokenize(const std::string &text, const bool add_bos = true);

  std::string detokenize(const std::vector<int> &tokens);

  std::string generate(const std::string &prompt, size_t max_tokens,
                       size_t max_generated_tokens, float temperature,
                       uint_fast32_t seed, int verbosity);
};