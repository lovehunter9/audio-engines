#include <fstream>
#include <iostream>
#include <mutex>
#include <signal.h>

static thread_local std::istream * s2st_input = nullptr;
std::istream & audio_s2st_input() { return *s2st_input; }

#define main audio_s2st_original_main
#include "shared-cli.cpp"
#undef main

extern "C" int audio_s2st_run(int argc, char ** argv, const char * pcm_path,
                              const char * events, const char * release) {
    static std::mutex serial;
    std::lock_guard<std::mutex> guard(serial);
    std::ifstream input(pcm_path, std::ios::binary);
    if (!input) return -10;
    s2st_input = &input;
    setenv("BENCHMARK_EVENTS_PATH", events, 1);
    setenv("BENCHMARK_HOLD_EXIT", release, 1);
    setenv("BENCHMARK_REALTIME_INPUT", "0", 1);
    g_is_generating = false;
    g_is_interrupted = false;
    struct sigaction previous;
    sigaction(SIGINT, nullptr, &previous);
    int result = -11;
    try { result = audio_s2st_original_main(argc, argv); }
    catch (const std::exception & error) { std::cerr << error.what() << std::endl; }
    sigaction(SIGINT, &previous, nullptr);
    s2st_input = nullptr;
    return result;
}
