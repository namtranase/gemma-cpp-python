#pragma once
// Command line text interface to gemma.

#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <vector>

// copybara:import_next_line:gemma_cpp
#include "compression/compress.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "gemma.h"  // Gemma
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/app.h"
// copybara:end
// copybara:import_next_line:gemma_cpp
#include "util/args.h"  // HasHelp
// copybara:end
#include "hwy/base.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/highway.h"
#include "hwy/per_target.h"
#include "hwy/profiler.h"
#include "hwy/timer.h"

using namespace gcpp;

class GemmaWrapper {
    public:
      // GemmaWrapper();
      void loadModel(const std::vector<std::string> &args); // Consider exception safety
      void showConfig();
      void showHelp();
      std::string completionPrompt();

    private:
        gcpp::LoaderArgs m_loader = gcpp::LoaderArgs(0, nullptr);
        gcpp::InferenceArgs m_inference = gcpp::InferenceArgs(0, nullptr);
        gcpp::AppArgs m_app = gcpp::AppArgs(0, nullptr);
        std::unique_ptr<gcpp::Gemma> m_model;
        KVCache m_kvcache;
};
