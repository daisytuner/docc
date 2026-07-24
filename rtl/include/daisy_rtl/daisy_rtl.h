#pragma once

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
#include <cstddef>
extern "C" {
#endif

enum __daisy_event_set {
    __DAISY_EVENT_SET_CPU = 0,
    __DAISY_EVENT_SET_CUDA = 1,
    __DAISY_EVENT_SET_NONE = 2,
};

typedef struct __daisy_metadata {
    // Source location
    const char* file_name;
    const char* function_name;
    long line_begin;
    long line_end;
    long column_begin;
    long column_end;

    // Docc metadata

    // SDFG-scope
    const char* sdfg_name;
    const char* sdfg_file;
    const char* arg_capture_path;
    const char* features_file;
    const char* opt_report_file;

    // Element-scope
    size_t element_id;
    const char* element_type;
    const char* target_type;

    // Loop Info
    int loopnest_index;
    size_t num_loops;
    size_t num_maps;
    size_t num_fors;
    size_t num_whiles;
    size_t max_depth;
    bool is_perfectly_nested;
    bool is_perfectly_parallel;
    bool is_elementwise;
    bool has_side_effects;

    // Example: sdfg_name + element_id
    const char* region_uuid;
} __daisy_metadata_t;

// Registers a region and returns a region ID
size_t __daisy_instrumentation_init(__daisy_metadata_t* metadata, enum __daisy_event_set event_set);

// Finalizes a region
void __daisy_instrumentation_finalize(size_t region_id);

// Enter a region, starts measurement
void __daisy_instrumentation_enter(size_t region_id);

// Exit a region, stops measurement and saves data
void __daisy_instrumentation_exit(size_t region_id);

// Increments a counter with name statically
// The counters appear with the "static:::" prefix in the output
void __daisy_instrumentation_increment(size_t region_id, const char* name, long long value);
void __daisy_instrumentation_metric(size_t region_id, const char* name, double value);

// Read a region's running aggregate runtime stats (only populated in
// __DAISY_INSTRUMENTATION_MODE=aggregate). ``mean_us`` and ``variance_us2`` are
// in microseconds / microseconds^2. Returns false if the region has no samples
// yet. Lets an in-process harness decide when to stop dynamic sampling.
bool __daisy_instrumentation_stats(size_t region_id, double* mean_us, double* variance_us2, long long* count);

// Aggregate running runtime stats over ALL regions (aggregate mode): per-iteration
// mean is the sum of the regions' means (mirrors the trace's summed durations),
// variance the sum of variances, count the min sample count across regions.
// Returns false if no region has samples yet. For the in-harness sampler.
bool __daisy_instrumentation_total_stats(double* mean_us, double* variance_us2, long long* count);

// Reset all regions' running aggregate stats (e.g. to drop warmup samples
// before the timed sampling loop). Leaves region registration intact.
void __daisy_instrumentation_reset_all(void);

typedef struct __daisy_capture __daisy_capture_t;

__daisy_capture_t* __daisy_capture_init(const char* name, const char* base_dir);

bool __daisy_capture_enter(__daisy_capture_t* context, size_t element_id);

void __daisy_capture_end(__daisy_capture_t* context);

void __daisy_capture_raw(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    bool after,
    size_t element_id
);

void __daisy_capture_1d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_elements,
    bool after,
    size_t element_id
);

void __daisy_capture_2d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_rows,
    size_t num_cols,
    bool after,
    size_t element_id
);

void __daisy_capture_3d(
    __daisy_capture_t* context,
    int arg_idx,
    const void* data,
    size_t size,
    int primitive_type,
    size_t num_x,
    size_t num_y,
    size_t num_z,
    bool after,
    size_t element_id
);

#ifdef __cplusplus
}
#endif

#define __daisy_min(a, b) ((a) < (b) ? (a) : (b))
#define __daisy_max(a, b) ((a) > (b) ? (a) : (b))
#define __daisy_fma(a, b, c) a* b + c

// Implementation of integer functions for symbolic expressions
inline int __daisy_sym_pow(int base, int exp) {
    if (exp < 0) {
        return 0;
    }

    int result = 1;
    while (exp) {
        if (exp & 1) {
            result *= base;
        }
        exp >>= 1;
        base *= base;
    }
    return result;
}

#ifdef __DAISY_NVVM__

// type conversion
#define __daisy_d2i_hi __nvvm_d2i_hi
#define __daisy_d2i_lo __nvvm_d2i_lo
#define __daisy_lohi_i2d __nvvm_lohi_i2d

#define __daisy_d2i_rn __nvvm_d2i_rn
#define __daisy_d2i_rm __nvvm_d2i_rm
#define __daisy_d2i_rp __nvvm_d2i_rp
#define __daisy_d2i_rz __nvvm_d2i_rz

#define __daisy_i2d_rn __nvvm_i2d_rn
#define __daisy_i2d_rm __nvvm_i2d_rm
#define __daisy_i2d_rp __nvvm_i2d_rp
#define __daisy_i2d_rz __nvvm_i2d_rz

#define __daisy_d2f_rn __nvvm_d2f_rn
#define __daisy_d2f_rm __nvvm_d2f_rm
#define __daisy_d2f_rp __nvvm_d2f_rp
#define __daisy_d2f_rz __nvvm_d2f_rz

#define __daisy_d2ui_rn __nvvm_d2ui_rn
#define __daisy_d2ui_rm __nvvm_d2ui_rm
#define __daisy_d2ui_rp __nvvm_d2ui_rp
#define __daisy_d2ui_rz __nvvm_d2ui_rz

#define __daisy_ui2d_rn __nvvm_ui2d_rn
#define __daisy_ui2d_rm __nvvm_ui2d_rm
#define __daisy_ui2d_rp __nvvm_ui2d_rp
#define __daisy_ui2d_rz __nvvm_ui2d_rz

#define __daisy_d2ll_rn __nvvm_d2ll_rn
#define __daisy_d2ll_rm __nvvm_d2ll_rm
#define __daisy_d2ll_rp __nvvm_d2ll_rp
#define __daisy_d2ll_rz __nvvm_d2ll_rz

#define __daisy_ll2d_rn __nvvm_ll2d_rn
#define __daisy_ll2d_rm __nvvm_ll2d_rm
#define __daisy_ll2d_rp __nvvm_ll2d_rp
#define __daisy_ll2d_rz __nvvm_ll2d_rz

#define __daisy_d2ull_rn __nvvm_d2ull_rn
#define __daisy_d2ull_rm __nvvm_d2ull_rm
#define __daisy_d2ull_rp __nvvm_d2ull_rp
#define __daisy_d2ull_rz __nvvm_d2ull_rz

#define __daisy_ull2d_rn __nvvm_ull2d_rn
#define __daisy_ull2d_rm __nvvm_ull2d_rm
#define __daisy_ull2d_rp __nvvm_ull2d_rp
#define __daisy_ull2d_rz __nvvm_ull2d_rz

#define __daisy_f2i_rn __nvvm_f2i_rn
#define __daisy_f2i_rm __nvvm_f2i_rm
#define __daisy_f2i_rp __nvvm_f2i_rp
#define __daisy_f2i_rz __nvvm_f2i_rz

#define __daisy_i2f_rn __nvvm_i2f_rn
#define __daisy_i2f_rm __nvvm_i2f_rm
#define __daisy_i2f_rp __nvvm_i2f_rp
#define __daisy_i2f_rz __nvvm_i2f_rz

#define __daisy_f2ui_rn __nvvm_f2ui_rn
#define __daisy_f2ui_rm __nvvm_f2ui_rm
#define __daisy_f2ui_rp __nvvm_f2ui_rp
#define __daisy_f2ui_rz __nvvm_f2ui_rz

#define __daisy_ui2f_rn __nvvm_ui2f_rn
#define __daisy_ui2f_rm __nvvm_ui2f_rm
#define __daisy_ui2f_rp __nvvm_ui2f_rp
#define __daisy_ui2f_rz __nvvm_ui2f_rz

#define __daisy_f2ll_rn __nvvm_f2ll_rn
#define __daisy_f2ll_rm __nvvm_f2ll_rm
#define __daisy_f2ll_rp __nvvm_f2ll_rp
#define __daisy_f2ll_rz __nvvm_f2ll_rz

#define __daisy_ll2f_rn __nvvm_ll2f_rn
#define __daisy_ll2f_rm __nvvm_ll2f_rm
#define __daisy_ll2f_rp __nvvm_ll2f_rp
#define __daisy_ll2f_rz __nvvm_ll2f_rz

#define __daisy_f2ull_rn __nvvm_f2ull_rn
#define __daisy_f2ull_rm __nvvm_f2ull_rm
#define __daisy_f2ull_rp __nvvm_f2ull_rp
#define __daisy_f2ull_rz __nvvm_f2ull_rz

#define __daisy_ull2f_rn __nvvm_ull2f_rn
#define __daisy_ull2f_rm __nvvm_ull2f_rm
#define __daisy_ull2f_rp __nvvm_ull2f_rp
#define __daisy_ull2f_rz __nvvm_ull2f_rz

#define __daisy_f2bf16_rn __nvvm_f2bf16_rn
#define __daisy_f2bf16_rz __nvvm_f2bf16_rz

#define __daisy_f2h_rn __nvvm_f2h_rn

// saturate
#define __daisy_saturate_f __nvvm_saturate_f
#define __daisy_saturate_d __nvvm_saturate_d

// fma instructions
#define __daisy_fma_rn_f __nvvm_fma_rn_f
#define __daisy_fma_rn_d __nvvm_fma_rn_d

#define __daisy_fma_rm_f __nvvm_fma_rm_f
#define __daisy_fma_rm_d __nvvm_fma_rm_d

#define __daisy_fma_rp_f __nvvm_fma_rp_f
#define __daisy_fma_rp_d __nvvm_fma_rp_d

#define __daisy_fma_rz_f __nvvm_fma_rz_f
#define __daisy_fma_rz_d __nvvm_fma_rz_d

#define __daisy_fma_rn_ftz_f __nvvm_fma_rn_ftz_f

#define __daisy_fma_rm_ftz_f __nvvm_fma_rm_ftz_f

#define __daisy_fma_rp_ftz_f __nvvm_fma_rp_ftz_f

#define __daisy_fma_rz_ftz_f __nvvm_fma_rz_ftz_f

#endif // __DAISY_NVVM__
