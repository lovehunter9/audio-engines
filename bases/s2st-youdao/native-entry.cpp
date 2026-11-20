// Embed the unmodified official server implementation without replacing Python signals.
#include "arg.h"
#include "common.h"
#include "llama.h"
#include <clocale>
#include <mutex>

int llama_server(common_params &, int, char **);
void llama_server_terminate();
void server_stream_session_manager_start();
void server_stream_session_manager_stop();

extern "C" int audio_t3po_run(int argc, char ** argv) {
    static std::mutex lifetime;
    std::lock_guard<std::mutex> lock(lifetime);
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    server_stream_session_manager_start();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        server_stream_session_manager_stop();
        return 1;
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    // The official CLI overload owns the same server loop and cleanup; nullptr
    // selects its existing embedding path, which leaves process signals alone.
    return llama_server(params, 0, nullptr);
}

extern "C" void audio_t3po_stop() { llama_server_terminate(); }
