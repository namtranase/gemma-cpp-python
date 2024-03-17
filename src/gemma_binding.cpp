#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "gemma_binding.h"
namespace py = pybind11;

static constexpr std::string_view kAsciiArtBanner =
    "  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __\n"
    " / _` |/ _ \\ '_ ` _ \\| '_ ` _ \\ / _` | / __| '_ \\| '_ \\\n"
    "| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |\n"
    " \\__, |\\___|_| |_| |_|_| |_| |_|\\__,_(_)___| .__/| .__/\n"
    "  __/ |                                    | |   | |\n"
    " |___/                                     |_|   |_|";

void ShowConfig(LoaderArgs& loader, InferenceArgs& inference, AppArgs& app) {
  loader.Print(app.verbosity);
  inference.Print(app.verbosity);
  app.Print(app.verbosity);

  if (app.verbosity >= 2) {
    time_t now = time(nullptr);
    char* dt = ctime(&now);  // NOLINT
    std::cout << "Date & Time                   : " << dt
              << "Prefill Token Batch Size      : " << gcpp::kPrefillBatchSize
              << "\n"
              << "Hardware concurrency          : "
              << std::thread::hardware_concurrency() << std::endl
              << "Instruction set               : "
              << hwy::TargetName(hwy::DispatchedTarget()) << " ("
              << hwy::VectorBytes() * 8 << " bits)" << "\n"
              << "Compiled config               : " << CompiledConfig() << "\n"
              << "Weight Type                   : "
              << gcpp::TypeName(gcpp::WeightT()) << "\n"
              << "EmbedderInput Type            : "
              << gcpp::TypeName(gcpp::EmbedderInputT()) << "\n";
  }
}

void ShowHelp(gcpp::LoaderArgs& loader, gcpp::InferenceArgs& inference,
              gcpp::AppArgs& app) {
  std::cerr
      << kAsciiArtBanner
      << "\n\ngemma.cpp : a lightweight, standalone C++ inference engine\n"
         "==========================================================\n\n"
         "To run gemma.cpp, you need to "
         "specify 3 required model loading arguments:\n    --tokenizer\n    "
         "--compressed_weights\n"
         "    --model.\n";
  std::cerr << "\n*Example Usage*\n\n./gemma --tokenizer tokenizer.spm "
               "--compressed_weights 2b-it-sfp.sbs --model 2b-it\n";
  std::cerr << "\n*Model Loading Arguments*\n\n";
  loader.Help();
  std::cerr << "\n*Inference Arguments*\n\n";
  inference.Help();
  std::cerr << "\n*Application Arguments*\n\n";
  app.Help();
  std::cerr << "\n";
}

void ReplGemma(gcpp::Gemma& model, gcpp::KVCache& kv_cache,
               hwy::ThreadPool& pool, hwy::ThreadPool& inner_pool,
               const InferenceArgs& args, int verbosity,
               const gcpp::AcceptFunc& accept_token, std::string& eot_line) {
  PROFILER_ZONE("Gen.misc");
  int abs_pos = 0;      // absolute token index over all turns
  int current_pos = 0;  // token index within the current turn
  int prompt_size{};

  std::mt19937 gen;
  if (args.deterministic) {
    gen.seed(42);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  // callback function invoked for each generated token.
  auto stream_token = [&abs_pos, &current_pos, &args, &gen, &prompt_size,
                       tokenizer = model.Tokenizer(),
                       verbosity](int token, float) {
    ++abs_pos;
    ++current_pos;
    if (current_pos < prompt_size) {
      std::cerr << "." << std::flush;
    } else if (token == gcpp::EOS_ID) {
      if (!args.multiturn) {
        abs_pos = 0;
        if (args.deterministic) {
          gen.seed(42);
        }
      }
      if (verbosity >= 2) {
        std::cout << "\n[ End ]\n";
      }
    } else {
      std::string token_text;
      HWY_ASSERT(tokenizer->Decode(std::vector<int>{token}, &token_text).ok());
      // +1 since position is incremented above
      if (current_pos == prompt_size + 1) {
        // first token of response
        token_text.erase(0, token_text.find_first_not_of(" \t\n"));
        if (verbosity >= 1) {
          std::cout << std::endl << std::endl;
        }
      }
      std::cout << token_text << std::flush;
    }
    return true;
  };

  while (abs_pos < args.max_tokens) {
    std::string prompt_string;
    std::vector<int> prompt;
    current_pos = 0;
    {
      PROFILER_ZONE("Gen.input");
      if (verbosity >= 1) {
        std::cout << "> " << std::flush;
      }

      if (eot_line.size() == 0) {
        std::getline(std::cin, prompt_string);
      } else {
        std::string line;
        while (std::getline(std::cin, line)) {
          if (line == eot_line) {
            break;
          }
          prompt_string += line + "\n";
        }
      }
    }

    if (std::cin.fail() || prompt_string == "%q" || prompt_string == "%Q") {
      return;
    }

    if (prompt_string == "%c" || prompt_string == "%C") {
      abs_pos = 0;
      continue;
    }

    if (model.model_training == ModelTraining::GEMMA_IT) {
      // For instruction-tuned models: add control tokens.
      prompt_string = "<start_of_turn>user\n" + prompt_string +
                      "<end_of_turn>\n<start_of_turn>model\n";
      if (abs_pos > 0) {
        // Prepend "<end_of_turn>" token if this is a multi-turn dialogue
        // continuation.
        prompt_string = "<end_of_turn>\n" + prompt_string;
      }
    }

    HWY_ASSERT(model.Tokenizer()->Encode(prompt_string, &prompt).ok());

    // For both pre-trained and instruction-tuned models: prepend "<bos>" token
    // if needed.
    if (abs_pos == 0) {
      prompt.insert(prompt.begin(), 2);
    }

    prompt_size = prompt.size();

    std::cerr << std::endl << "[ Reading prompt ] " << std::flush;

    const double time_start = hwy::platform::Now();
    GenerateGemma(model, args.max_tokens, args.max_generated_tokens,
                  args.temperature, prompt, abs_pos, kv_cache, pool, inner_pool,
                  stream_token, accept_token, gen, verbosity);
    const double time_end = hwy::platform::Now();
    const double tok_sec = current_pos / (time_end - time_start);
    if (verbosity >= 2) {
      std::cout << current_pos << " tokens (" << abs_pos << " total tokens)"
                << std::endl
                << tok_sec << " tokens / sec" << std::endl;
    }
    std::cout << std::endl << std::endl;
  }
  std::cout
      << "max_tokens (" << args.max_tokens
      << ") exceeded. Use a larger value if desired using the --max_tokens "
      << "command line flag.\n";
}

void GemmaWrapper::loadModel(const std::vector<std::string> &args) {
  int argc = args.size() + 1; // +1 for the program name
  std::vector<char *> argv_vec;
  argv_vec.reserve(argc);
  argv_vec.push_back(const_cast<char *>("pygemma"));
  for (const auto &arg : args)
  {
      argv_vec.push_back(const_cast<char *>(arg.c_str()));
  }

  char **argv = argv_vec.data();

  this->m_loader = gcpp::LoaderArgs(argc, argv);
  this->m_inference = gcpp::InferenceArgs(argc, argv);
  this->m_app = gcpp::AppArgs(argc, argv);

  PROFILER_ZONE("Run.misc");

  hwy::ThreadPool inner_pool(0);
  hwy::ThreadPool pool(this->m_app.num_threads);
  // For many-core, pinning threads to cores helps.
  if (this->m_app.num_threads > 10) {
    PinThreadToCore(this->m_app.num_threads - 1);  // Main thread

    pool.Run(0, pool.NumThreads(),
             [](uint64_t /*task*/, size_t thread) { PinThreadToCore(thread); });
  }

  if (!this->m_model) {
    this->m_model.reset(new gcpp::Gemma(this->m_loader.tokenizer, this->m_loader.compressed_weights, this->m_loader.ModelType(), pool));
  }
//   auto kvcache = CreateKVCache(loader.ModelType());
  this->m_kvcache = CreateKVCache(this->m_loader.ModelType());

  if (const char* error = this->m_inference.Validate()) {
    ShowHelp(this->m_loader, this->m_inference, this->m_app);
    HWY_ABORT("\nInvalid args: %s", error);
  }

  if (this->m_app.verbosity >= 1) {
    const std::string instructions =
        "*Usage*\n"
        "  Enter an instruction and press enter (%C resets conversation, "
        "%Q quits).\n" +
        (this->m_inference.multiturn == 0
             ? std::string("  Since multiturn is set to 0, conversation will "
                           "automatically reset every turn.\n\n")
             : "\n") +
        "*Examples*\n"
        "  - Write an email to grandma thanking her for the cookies.\n"
        "  - What are some historical attractions to visit around "
        "Massachusetts?\n"
        "  - Compute the nth fibonacci number in javascript.\n"
        "  - Write a standup comedy bit about GPU programming.\n";

    std::cout << "\033[2J\033[1;1H"  // clear screen
              << kAsciiArtBanner << "\n\n";
    ShowConfig(this->m_loader, this->m_inference, this->m_app);
    std::cout << "\n" << instructions << "\n";
  }
}

void GemmaWrapper::showConfig() {
    ShowConfig(this->m_loader,this->m_inference, this->m_app);
}

void GemmaWrapper::showHelp() {
    ShowHelp(this->m_loader,this->m_inference, this->m_app);
}


PYBIND11_MODULE(pygemma, m) {
    py::class_<GemmaWrapper>(m, "Gemma")
        .def(py::init<>())
        .def("show_config", &GemmaWrapper::showConfig)
        .def("show_help",  &GemmaWrapper::showHelp)
        .def("load_model", [](GemmaWrapper &self,
                              const std::string &tokenizer,
                              const std::string &compressed_weights,
                              const std::string &model) {
            std::vector<std::string> args = {
                "--tokenizer", tokenizer,
                "--compressed_weights", compressed_weights,
                "--model", model
            };
            self.loadModel(args); // Assuming GemmaWrapper::loadModel accepts std::vector<std::string>
        }, py::arg("tokenizer"), py::arg("compressed_weights"), py::arg("model"))
        .def("completion", &GemmaWrapper::completionPrompt);
}
