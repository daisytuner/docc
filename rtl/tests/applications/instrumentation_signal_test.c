#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <daisy_rtl/daisy_rtl.h>

// Runs three region invocations, then ends the process without a normal exit:
//   sigterm         - terminated by SIGTERM
//   sigterm_active  - terminated by SIGTERM while the region is entered
//   sigterm_handler - SIGTERM handled by an application handler installed before the RTL
//   dump_exit       - explicit dump, then _exit without running destructors

static void application_handler(int sig) {
    const char message[] = "application handler\n";
    write(STDOUT_FILENO, message, sizeof(message) - 1);
    _exit(42);
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "sigterm";
    if (strcmp(mode, "sigterm_handler") == 0) {
        signal(SIGTERM, application_handler);
    }

    __daisy_metadata_t metadata = {
        .file_name = "instrumentation_signal_test.c",
        .function_name = "main",
        .line_begin = 30,
        .line_end = 40,
        .column_begin = 4,
        .column_end = 5,
        .sdfg_name = "__daisy_instrumentation_signal_test_0",
        .sdfg_file = "/tmp/DOCC/0000-0000/123456789/sdfg_0.json",
        .element_id = 10,
        .element_type = "for",
        .target_type = "sequential",
        .loopnest_index = 0,
        .region_uuid = "__daisy_instrumentation_signal_test_0_10"
    };
    size_t region_id = __daisy_instrumentation_init(&metadata, __DAISY_EVENT_SET_CPU);

    volatile double sum = 0.0;
    for (int rep = 0; rep < 3; rep++) {
        __daisy_instrumentation_enter(region_id);
        for (int i = 0; i < 1000; i++) {
            sum += i;
        }
        __daisy_instrumentation_exit(region_id);
    }

    if (strcmp(mode, "dump_exit") == 0) {
        _exit(__daisy_instrumentation_dump() ? 0 : 1);
    }
    if (strcmp(mode, "sigterm_active") == 0) {
        __daisy_instrumentation_enter(region_id);
    }
    raise(SIGTERM);
    return 1;
}
