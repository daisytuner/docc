#include "docc/lifting/functions/function_lifting.h"

#include <sdfg/data_flow/library_nodes/math/math.h>

namespace docc {
namespace lifting {

class LibFuncLifting : public FunctionLifting {
private:
    sdfg::control_flow::State& visit_tasklet(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_math(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_calloc(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_free(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

    sdfg::control_flow::State& visit_malloc(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    );

    static bool is_tasklet(llvm::Intrinsic::ID iid) {
        switch (iid) {
            case llvm::LibFunc_abs:
            case llvm::LibFunc_labs:
            case llvm::LibFunc_llabs:
                return true;
            default:
                return false;
        }
    }

    static sdfg::data_flow::TaskletCode as_tasklet(llvm::Intrinsic::ID iid) {
        switch (iid) {
            case llvm::Intrinsic::expect:
            case llvm::LibFunc_abs:
            case llvm::LibFunc_labs:
            case llvm::LibFunc_llabs:
                return sdfg::data_flow::TaskletCode::int_abs;
            default:
                throw std::runtime_error("Intrinsic not mapped to a tasklet");
        }
    }

    static bool is_math(llvm::LibFunc lf) {
        switch (lf) {
            case llvm::LibFunc_acos_finite:
            case llvm::LibFunc_acosf_finite:
            case llvm::LibFunc_acosh_finite:
            case llvm::LibFunc_acoshf_finite:
            case llvm::LibFunc_acoshl_finite:
            case llvm::LibFunc_acosl_finite:
            case llvm::LibFunc_asin_finite:
            case llvm::LibFunc_asinf_finite:
            case llvm::LibFunc_asinl_finite:
            case llvm::LibFunc_atan2_finite:
            case llvm::LibFunc_atan2f_finite:
            case llvm::LibFunc_atan2l_finite:
            case llvm::LibFunc_atanh_finite:
            case llvm::LibFunc_atanhf_finite:
            case llvm::LibFunc_atanhl_finite:
            case llvm::LibFunc_cosh_finite:
            case llvm::LibFunc_coshf_finite:
            case llvm::LibFunc_coshl_finite:
            case llvm::LibFunc_exp10_finite:
            case llvm::LibFunc_exp10f_finite:
            case llvm::LibFunc_exp10l_finite:
            case llvm::LibFunc_exp2_finite:
            case llvm::LibFunc_exp2f_finite:
            case llvm::LibFunc_exp2l_finite:
            case llvm::LibFunc_exp_finite:
            case llvm::LibFunc_expf_finite:
            case llvm::LibFunc_expl_finite:
            case llvm::LibFunc_log10_finite:
            case llvm::LibFunc_log10f_finite:
            case llvm::LibFunc_log10l_finite:
            case llvm::LibFunc_log2_finite:
            case llvm::LibFunc_log2f_finite:
            case llvm::LibFunc_log2l_finite:
            case llvm::LibFunc_log_finite:
            case llvm::LibFunc_logf_finite:
            case llvm::LibFunc_logl_finite:
            case llvm::LibFunc_pow_finite:
            case llvm::LibFunc_powf_finite:
            case llvm::LibFunc_powl_finite:
            case llvm::LibFunc_sinh_finite:
            case llvm::LibFunc_sinhf_finite:
            case llvm::LibFunc_sinhl_finite:
            case llvm::LibFunc_sqrt_finite:
            case llvm::LibFunc_sqrtf_finite:
            case llvm::LibFunc_sqrtl_finite:
            case llvm::LibFunc_acos:
            case llvm::LibFunc_acosf:
            case llvm::LibFunc_acosh:
            case llvm::LibFunc_acoshf:
            case llvm::LibFunc_acoshl:
            case llvm::LibFunc_acosl:
            case llvm::LibFunc_asin:
            case llvm::LibFunc_asinf:
            case llvm::LibFunc_asinh:
            case llvm::LibFunc_asinhf:
            case llvm::LibFunc_asinhl:
            case llvm::LibFunc_asinl:
            case llvm::LibFunc_atan:
            case llvm::LibFunc_atan2:
            case llvm::LibFunc_atan2f:
            case llvm::LibFunc_atan2l:
            case llvm::LibFunc_atanf:
            case llvm::LibFunc_atanh:
            case llvm::LibFunc_atanhf:
            case llvm::LibFunc_atanhl:
            case llvm::LibFunc_atanl:
            case llvm::LibFunc_cos:
            case llvm::LibFunc_cosf:
            case llvm::LibFunc_cosh:
            case llvm::LibFunc_coshf:
            case llvm::LibFunc_coshl:
            case llvm::LibFunc_cosl:
            case llvm::LibFunc_exp:
            case llvm::LibFunc_exp10:
            case llvm::LibFunc_exp10f:
            case llvm::LibFunc_exp10l:
            case llvm::LibFunc_exp2:
            case llvm::LibFunc_exp2f:
            case llvm::LibFunc_exp2l:
            case llvm::LibFunc_expf:
            case llvm::LibFunc_expl:
            case llvm::LibFunc_expm1:
            case llvm::LibFunc_expm1f:
            case llvm::LibFunc_expm1l:
            case llvm::LibFunc_fabs:
            case llvm::LibFunc_fabsf:
            case llvm::LibFunc_fabsl:
            case llvm::LibFunc_floor:
            case llvm::LibFunc_floorf:
            case llvm::LibFunc_floorl:
            case llvm::LibFunc_fmax:
            case llvm::LibFunc_fmaxf:
            case llvm::LibFunc_fmaxl:
            case llvm::LibFunc_fmin:
            case llvm::LibFunc_fminf:
            case llvm::LibFunc_fminl:
            case llvm::LibFunc_fmod:
            case llvm::LibFunc_fmodf:
            case llvm::LibFunc_fmodl:
            case llvm::LibFunc_ldexp:
            case llvm::LibFunc_ldexpf:
            case llvm::LibFunc_ldexpl:
            case llvm::LibFunc_log:
            case llvm::LibFunc_log10:
            case llvm::LibFunc_log10f:
            case llvm::LibFunc_log10l:
            case llvm::LibFunc_log1p:
            case llvm::LibFunc_log1pf:
            case llvm::LibFunc_log1pl:
            case llvm::LibFunc_log2:
            case llvm::LibFunc_log2f:
            case llvm::LibFunc_log2l:
            case llvm::LibFunc_logb:
            case llvm::LibFunc_logbf:
            case llvm::LibFunc_logbl:
            case llvm::LibFunc_logf:
            case llvm::LibFunc_logl:
            case llvm::LibFunc_nearbyint:
            case llvm::LibFunc_nearbyintf:
            case llvm::LibFunc_nearbyintl:
            case llvm::LibFunc_pow:
            case llvm::LibFunc_powf:
            case llvm::LibFunc_powl:
            case llvm::LibFunc_rint:
            case llvm::LibFunc_rintf:
            case llvm::LibFunc_rintl:
            case llvm::LibFunc_round:
            case llvm::LibFunc_roundeven:
            case llvm::LibFunc_roundevenf:
            case llvm::LibFunc_roundevenl:
            case llvm::LibFunc_roundf:
            case llvm::LibFunc_roundl:
            case llvm::LibFunc_sin:
            case llvm::LibFunc_sinf:
            case llvm::LibFunc_sinh:
            case llvm::LibFunc_sinhf:
            case llvm::LibFunc_sinhl:
            case llvm::LibFunc_sinl:
            case llvm::LibFunc_sqrt:
            case llvm::LibFunc_sqrtf:
            case llvm::LibFunc_sqrtl:
            case llvm::LibFunc_tan:
            case llvm::LibFunc_tanf:
            case llvm::LibFunc_tanh:
            case llvm::LibFunc_tanhf:
            case llvm::LibFunc_tanhl:
            case llvm::LibFunc_tanl:
                return true;
            default:
                return false;
        }
    }

    static sdfg::math::cmath::CMathFunction as_math_function(llvm::LibFunc lf) {
        switch (lf) {
            // case llvm::LibFunc_acos_finite:
            //     return "__acos_finite";
            // case llvm::LibFunc_acosf_finite:
            //     return "__acosf_finite";
            // case llvm::LibFunc_acosh_finite:
            //     return "__acosh_finite";
            // case llvm::LibFunc_acoshf_finite:
            //     return "__acoshf_finite";
            // case llvm::LibFunc_acoshl_finite:
            //     return "__acoshl_finite";
            // case llvm::LibFunc_acosl_finite:
            //     return "__acosl_finite";
            // case llvm::LibFunc_asin_finite:
            //     return "__asin_finite";
            // case llvm::LibFunc_asinf_finite:
            //     return "__asinf_finite";
            // case llvm::LibFunc_asinl_finite:
            //     return "__asinl_finite";
            // case llvm::LibFunc_atan2_finite:
            //     return "__atan2_finite";
            // case llvm::LibFunc_atan2f_finite:
            //     return "__atan2f_finite";
            // case llvm::LibFunc_atan2l_finite:
            //     return "__atan2l_finite";
            // case llvm::LibFunc_atanh_finite:
            //     return "__atanh_finite";
            // case llvm::LibFunc_atanhf_finite:
            //     return "__atanhf_finite";
            // case llvm::LibFunc_atanhl_finite:
            //     return "__atanhl_finite";
            // case llvm::LibFunc_cosh_finite:
            //     return "__cosh_finite";
            // case llvm::LibFunc_coshf_finite:
            //     return "__coshf_finite";
            // case llvm::LibFunc_coshl_finite:
            //     return "__coshl_finite";
            // case llvm::LibFunc_exp10_finite:
            //     return "__exp10_finite";
            // case llvm::LibFunc_exp10f_finite:
            //     return "__exp10f_finite";
            // case llvm::LibFunc_exp10l_finite:
            //     return "__exp10l_finite";
            // case llvm::LibFunc_exp2_finite:
            //     return "__exp2_finite";
            // case llvm::LibFunc_exp2f_finite:
            //     return "__exp2f_finite";
            // case llvm::LibFunc_exp2l_finite:
            //     return "__exp2l_finite";
            // case llvm::LibFunc_exp_finite:
            //     return "__exp_finite";
            // case llvm::LibFunc_expf_finite:
            //     return "__expf_finite";
            // case llvm::LibFunc_expl_finite:
            //     return "__expl_finite";
            // case llvm::LibFunc_log10_finite:
            //     return "__log10_finite";
            // case llvm::LibFunc_log10f_finite:
            //     return "__log10f_finite";
            // case llvm::LibFunc_log10l_finite:
            //     return "__log10l_finite";
            // case llvm::LibFunc_log2_finite:
            //     return "__log2_finite";
            // case llvm::LibFunc_log2f_finite:
            //     return "__log2f_finite";
            // case llvm::LibFunc_log2l_finite:
            //     return "__log2l_finite";
            // case llvm::LibFunc_log_finite:
            //     return "__log_finite";
            // case llvm::LibFunc_logf_finite:
            //     return "__logf_finite";
            // case llvm::LibFunc_logl_finite:
            //     return "__logl_finite";
            // case llvm::LibFunc_pow_finite:
            //     return "__pow_finite";
            // case llvm::LibFunc_powf_finite:
            //     return "__powf_finite";
            // case llvm::LibFunc_powl_finite:
            //     return "__powl_finite";
            // case llvm::LibFunc_sinh_finite:
            //     return "__sinh_finite";
            // case llvm::LibFunc_sinhf_finite:
            //     return "__sinhf_finite";
            // case llvm::LibFunc_sinhl_finite:
            //     return "__sinhl_finite";
            // case llvm::LibFunc_sqrt_finite:
            //     return "__sqrt_finite";
            // case llvm::LibFunc_sqrtf_finite:
            //     return "__sqrtf_finite";
            // case llvm::LibFunc_sqrtl_finite:
            //     return "__sqrtl_finite";
            case llvm::LibFunc_acos:
            case llvm::LibFunc_acosf:
            case llvm::LibFunc_acosl:
                return sdfg::math::cmath::CMathFunction::acos;
            case llvm::LibFunc_acosh:
            case llvm::LibFunc_acoshf:
            case llvm::LibFunc_acoshl:
                return sdfg::math::cmath::CMathFunction::acosh;
            case llvm::LibFunc_asin:
            case llvm::LibFunc_asinf:
            case llvm::LibFunc_asinl:
                return sdfg::math::cmath::CMathFunction::asin;
            case llvm::LibFunc_asinh:
            case llvm::LibFunc_asinhf:
            case llvm::LibFunc_asinhl:
                return sdfg::math::cmath::CMathFunction::asinh;
            case llvm::LibFunc_atan:
            case llvm::LibFunc_atanl:
            case llvm::LibFunc_atanf:
                return sdfg::math::cmath::CMathFunction::atan;
            case llvm::LibFunc_atan2:
            case llvm::LibFunc_atan2f:
            case llvm::LibFunc_atan2l:
                return sdfg::math::cmath::CMathFunction::atan2;
            case llvm::LibFunc_atanh:
            case llvm::LibFunc_atanhf:
            case llvm::LibFunc_atanhl:
                return sdfg::math::cmath::CMathFunction::atanh;
            case llvm::LibFunc_cos:
            case llvm::LibFunc_cosf:
            case llvm::LibFunc_cosl:
                return sdfg::math::cmath::CMathFunction::cos;
            case llvm::LibFunc_cosh:
            case llvm::LibFunc_coshf:
            case llvm::LibFunc_coshl:
                return sdfg::math::cmath::CMathFunction::cosh;
            case llvm::LibFunc_exp:
            case llvm::LibFunc_expf:
            case llvm::LibFunc_expl:
                return sdfg::math::cmath::CMathFunction::exp;
            case llvm::LibFunc_exp10:
            case llvm::LibFunc_exp10f:
            case llvm::LibFunc_exp10l:
                return sdfg::math::cmath::CMathFunction::exp10;
            case llvm::LibFunc_exp2:
            case llvm::LibFunc_exp2f:
            case llvm::LibFunc_exp2l:
                return sdfg::math::cmath::CMathFunction::exp2;
            case llvm::LibFunc_expm1:
            case llvm::LibFunc_expm1f:
            case llvm::LibFunc_expm1l:
                return sdfg::math::cmath::CMathFunction::expm1;
            case llvm::LibFunc_fabs:
            case llvm::LibFunc_fabsf:
            case llvm::LibFunc_fabsl:
                return sdfg::math::cmath::CMathFunction::fabs;
            case llvm::LibFunc_floor:
            case llvm::LibFunc_floorf:
            case llvm::LibFunc_floorl:
                return sdfg::math::cmath::CMathFunction::floor;
            case llvm::LibFunc_fmax:
            case llvm::LibFunc_fmaxf:
            case llvm::LibFunc_fmaxl:
                return sdfg::math::cmath::CMathFunction::fmax;
            case llvm::LibFunc_fmin:
            case llvm::LibFunc_fminf:
            case llvm::LibFunc_fminl:
                return sdfg::math::cmath::CMathFunction::fmin;
            case llvm::LibFunc_fmod:
            case llvm::LibFunc_fmodf:
            case llvm::LibFunc_fmodl:
                return sdfg::math::cmath::CMathFunction::fmod;
            case llvm::LibFunc_ldexp:
            case llvm::LibFunc_ldexpf:
            case llvm::LibFunc_ldexpl:
                return sdfg::math::cmath::CMathFunction::ldexp;
            case llvm::LibFunc_log:
            case llvm::LibFunc_logf:
            case llvm::LibFunc_logl:
                return sdfg::math::cmath::CMathFunction::log;
            case llvm::LibFunc_log10:
            case llvm::LibFunc_log10f:
            case llvm::LibFunc_log10l:
                return sdfg::math::cmath::CMathFunction::log10;
            case llvm::LibFunc_log1p:
            case llvm::LibFunc_log1pf:
            case llvm::LibFunc_log1pl:
                return sdfg::math::cmath::CMathFunction::log1p;
            case llvm::LibFunc_log2:
            case llvm::LibFunc_log2f:
            case llvm::LibFunc_log2l:
                return sdfg::math::cmath::CMathFunction::log2;
            case llvm::LibFunc_logb:
            case llvm::LibFunc_logbf:
            case llvm::LibFunc_logbl:
                return sdfg::math::cmath::CMathFunction::logb;
            case llvm::LibFunc_nearbyint:
            case llvm::LibFunc_nearbyintf:
            case llvm::LibFunc_nearbyintl:
                return sdfg::math::cmath::CMathFunction::nearbyint;
            case llvm::LibFunc_pow:
            case llvm::LibFunc_powf:
            case llvm::LibFunc_powl:
                return sdfg::math::cmath::CMathFunction::pow;
            case llvm::LibFunc_rint:
            case llvm::LibFunc_rintf:
            case llvm::LibFunc_rintl:
                return sdfg::math::cmath::CMathFunction::rint;
            case llvm::LibFunc_round:
            case llvm::LibFunc_roundf:
            case llvm::LibFunc_roundl:
                return sdfg::math::cmath::CMathFunction::round;
            case llvm::LibFunc_roundeven:
            case llvm::LibFunc_roundevenf:
            case llvm::LibFunc_roundevenl:
                return sdfg::math::cmath::CMathFunction::roundeven;
            case llvm::LibFunc_sin:
            case llvm::LibFunc_sinf:
            case llvm::LibFunc_sinl:
                return sdfg::math::cmath::CMathFunction::sin;
            case llvm::LibFunc_sinh:
            case llvm::LibFunc_sinhf:
            case llvm::LibFunc_sinhl:
                return sdfg::math::cmath::CMathFunction::sinh;
            case llvm::LibFunc_sqrt:
            case llvm::LibFunc_sqrtf:
            case llvm::LibFunc_sqrtl:
                return sdfg::math::cmath::CMathFunction::sqrt;
            case llvm::LibFunc_tan:
            case llvm::LibFunc_tanf:
            case llvm::LibFunc_tanl:
                return sdfg::math::cmath::CMathFunction::tan;
            case llvm::LibFunc_tanh:
            case llvm::LibFunc_tanhf:
            case llvm::LibFunc_tanhl:
                return sdfg::math::cmath::CMathFunction::tanh;
            default:
                throw std::runtime_error("libfunc not mapped to a function");
        }
    }

public:
    LibFuncLifting(
        llvm::TargetLibraryInfo& TLI,
        const llvm::DataLayout& DL,
        const llvm::Function& function,
        sdfg::FunctionType target_type,
        sdfg::builder::SDFGBuilder& builder,
        std::map<const llvm::BasicBlock*, std::set<const sdfg::control_flow::State*>>& state_mapping,
        std::map<const sdfg::control_flow::State*, std::set<const llvm::BasicBlock*>>& pred_mapping,
        std::unordered_map<const llvm::Value*, std::string>& constants_mapping,
        std::unordered_map<const llvm::Type*, std::string> anonymous_types_mapping
    )
        : FunctionLifting(
              TLI,
              DL,
              function,
              target_type,
              builder,
              state_mapping,
              pred_mapping,
              constants_mapping,
              anonymous_types_mapping
          ) {}

    sdfg::control_flow::State& visit(
        const llvm::BasicBlock* block, const llvm::CallBase* instruction, sdfg::control_flow::State& current_state
    ) override;

    static bool is_supported(llvm::LibFunc& lf) {
        switch (lf) {
            case llvm::LibFunc_msvc_new_int:
                return false;
            case llvm::LibFunc_msvc_new_int_nothrow:
                return false;
            case llvm::LibFunc_msvc_new_longlong:
                return false;
            case llvm::LibFunc_msvc_new_longlong_nothrow:
                return false;
            case llvm::LibFunc_msvc_delete_ptr32:
                return false;
            case llvm::LibFunc_msvc_delete_ptr32_nothrow:
                return false;
            case llvm::LibFunc_msvc_delete_ptr32_int:
                return false;
            case llvm::LibFunc_msvc_delete_ptr64:
                return false;
            case llvm::LibFunc_msvc_delete_ptr64_nothrow:
                return false;
            case llvm::LibFunc_msvc_delete_ptr64_longlong:
                return false;
            case llvm::LibFunc_msvc_new_array_int:
                return false;
            case llvm::LibFunc_msvc_new_array_int_nothrow:
                return false;
            case llvm::LibFunc_msvc_new_array_longlong:
                return false;
            case llvm::LibFunc_msvc_new_array_longlong_nothrow:
                return false;
            case llvm::LibFunc_msvc_delete_array_ptr32:
                return false;
            case llvm::LibFunc_msvc_delete_array_ptr32_nothrow:
                return false;
            case llvm::LibFunc_msvc_delete_array_ptr32_int:
                return false;
            case llvm::LibFunc_msvc_delete_array_ptr64:
                return false;
            case llvm::LibFunc_msvc_delete_array_ptr64_nothrow:
                return false;
            case llvm::LibFunc_msvc_delete_array_ptr64_longlong:
                return false;
            case llvm::LibFunc_under_IO_getc:
                return false;
            case llvm::LibFunc_under_IO_putc:
                return false;
            case llvm::LibFunc_ZdaPv:
                return false;
            case llvm::LibFunc_ZdaPvRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZdaPvSt11align_val_t:
                return false;
            case llvm::LibFunc_ZdaPvSt11align_val_tRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZdaPvj:
                return false;
            case llvm::LibFunc_ZdaPvjSt11align_val_t:
                return false;
            case llvm::LibFunc_ZdaPvm:
                return false;
            case llvm::LibFunc_ZdaPvmSt11align_val_t:
                return false;
            case llvm::LibFunc_ZdlPv:
                return false;
            case llvm::LibFunc_ZdlPvRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZdlPvSt11align_val_t:
                return false;
            case llvm::LibFunc_ZdlPvSt11align_val_tRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZdlPvj:
                return false;
            case llvm::LibFunc_ZdlPvjSt11align_val_t:
                return false;
            case llvm::LibFunc_ZdlPvm:
                return false;
            case llvm::LibFunc_ZdlPvmSt11align_val_t:
                return false;
            case llvm::LibFunc_Znaj:
                return false;
            case llvm::LibFunc_ZnajRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZnajSt11align_val_t:
                return false;
            case llvm::LibFunc_ZnajSt11align_val_tRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_Znam:
                return false;
            case llvm::LibFunc_Znam12__hot_cold_t:
                return false;
            case llvm::LibFunc_ZnamRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZnamRKSt9nothrow_t12__hot_cold_t:
                return false;
            case llvm::LibFunc_ZnamSt11align_val_t:
                return false;
            case llvm::LibFunc_ZnamSt11align_val_t12__hot_cold_t:
                return false;
            case llvm::LibFunc_ZnamSt11align_val_tRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZnamSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
                return false;
            case llvm::LibFunc_Znwj:
                return false;
            case llvm::LibFunc_ZnwjRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZnwjSt11align_val_t:
                return false;
            case llvm::LibFunc_ZnwjSt11align_val_tRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_Znwm:
                return false;
            case llvm::LibFunc_Znwm12__hot_cold_t:
                return false;
            case llvm::LibFunc_ZnwmRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZnwmRKSt9nothrow_t12__hot_cold_t:
                return false;
            case llvm::LibFunc_ZnwmSt11align_val_t:
                return false;
            case llvm::LibFunc_ZnwmSt11align_val_t12__hot_cold_t:
                return false;
            case llvm::LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t:
                return false;
            case llvm::LibFunc_ZnwmSt11align_val_tRKSt9nothrow_t12__hot_cold_t:
                return false;
            case llvm::LibFunc_acos_finite:
                return true;
            case llvm::LibFunc_acosf_finite:
                return true;
            case llvm::LibFunc_acosh_finite:
                return true;
            case llvm::LibFunc_acoshf_finite:
                return true;
            case llvm::LibFunc_acoshl_finite:
                return true;
            case llvm::LibFunc_acosl_finite:
                return true;
            case llvm::LibFunc_asin_finite:
                return true;
            case llvm::LibFunc_asinf_finite:
                return true;
            case llvm::LibFunc_asinl_finite:
                return true;
            case llvm::LibFunc_atan2_finite:
                return true;
            case llvm::LibFunc_atan2f_finite:
                return true;
            case llvm::LibFunc_atan2l_finite:
                return true;
            case llvm::LibFunc_atanh_finite:
                return true;
            case llvm::LibFunc_atanhf_finite:
                return true;
            case llvm::LibFunc_atanhl_finite:
                return true;
            case llvm::LibFunc_atomic_load:
                return false;
            case llvm::LibFunc_atomic_store:
                return false;
            case llvm::LibFunc_cosh_finite:
                return true;
            case llvm::LibFunc_coshf_finite:
                return true;
            case llvm::LibFunc_coshl_finite:
                return true;
            case llvm::LibFunc_cospi:
                return false;
            case llvm::LibFunc_cospif:
                return false;
            case llvm::LibFunc_cxa_atexit:
                return false;
            case llvm::LibFunc_atexit:
                return false;
            case llvm::LibFunc_cxa_guard_abort:
                return false;
            case llvm::LibFunc_cxa_guard_acquire:
                return false;
            case llvm::LibFunc_cxa_guard_release:
                return false;
            case llvm::LibFunc_exp10_finite:
                return true;
            case llvm::LibFunc_exp10f_finite:
                return true;
            case llvm::LibFunc_exp10l_finite:
                return true;
            case llvm::LibFunc_exp2_finite:
                return true;
            case llvm::LibFunc_exp2f_finite:
                return true;
            case llvm::LibFunc_exp2l_finite:
                return true;
            case llvm::LibFunc_exp_finite:
                return true;
            case llvm::LibFunc_expf_finite:
                return true;
            case llvm::LibFunc_expl_finite:
                return true;
            case llvm::LibFunc_dunder_isoc99_scanf:
                return false;
            case llvm::LibFunc_dunder_isoc99_sscanf:
                return false;
            case llvm::LibFunc___kmpc_alloc_shared:
                return false;
            case llvm::LibFunc___kmpc_free_shared:
                return false;
            case llvm::LibFunc_log10_finite:
                return true;
            case llvm::LibFunc_log10f_finite:
                return true;
            case llvm::LibFunc_log10l_finite:
                return true;
            case llvm::LibFunc_log2_finite:
                return true;
            case llvm::LibFunc_log2f_finite:
                return true;
            case llvm::LibFunc_log2l_finite:
                return true;
            case llvm::LibFunc_log_finite:
                return true;
            case llvm::LibFunc_logf_finite:
                return true;
            case llvm::LibFunc_logl_finite:
                return true;
            case llvm::LibFunc_memccpy_chk:
                return false;
            case llvm::LibFunc_memcpy_chk:
                return false;
            case llvm::LibFunc_memmove_chk:
                return false;
            case llvm::LibFunc_mempcpy_chk:
                return false;
            case llvm::LibFunc_memset_chk:
                return false;
            case llvm::LibFunc_nvvm_reflect:
                return false;
            case llvm::LibFunc_pow_finite:
                return true;
            case llvm::LibFunc_powf_finite:
                return true;
            case llvm::LibFunc_powl_finite:
                return true;
            case llvm::LibFunc_sincospi_stret:
                return false;
            case llvm::LibFunc_sincospif_stret:
                return false;
            case llvm::LibFunc_sinh_finite:
                return true;
            case llvm::LibFunc_sinhf_finite:
                return true;
            case llvm::LibFunc_sinhl_finite:
                return true;
            case llvm::LibFunc_sinpi:
                return false;
            case llvm::LibFunc_sinpif:
                return false;
            case llvm::LibFunc_small_fprintf:
                return false;
            case llvm::LibFunc_small_printf:
                return false;
            case llvm::LibFunc_small_sprintf:
                return false;
            case llvm::LibFunc_snprintf_chk:
                return false;
            case llvm::LibFunc_sprintf_chk:
                return false;
            case llvm::LibFunc_sqrt_finite:
                return true;
            case llvm::LibFunc_sqrtf_finite:
                return true;
            case llvm::LibFunc_sqrtl_finite:
                return true;
            case llvm::LibFunc_stpcpy_chk:
                return false;
            case llvm::LibFunc_stpncpy_chk:
                return false;
            case llvm::LibFunc_strcat_chk:
                return false;
            case llvm::LibFunc_strcpy_chk:
                return false;
            case llvm::LibFunc_dunder_strdup:
                return false;
            case llvm::LibFunc_strlcat_chk:
                return false;
            case llvm::LibFunc_strlcpy_chk:
                return false;
            case llvm::LibFunc_strlen_chk:
                return false;
            case llvm::LibFunc_strncat_chk:
                return false;
            case llvm::LibFunc_strncpy_chk:
                return false;
            case llvm::LibFunc_dunder_strndup:
                return false;
            case llvm::LibFunc_dunder_strtok_r:
                return false;
            case llvm::LibFunc_vsnprintf_chk:
                return false;
            case llvm::LibFunc_vsprintf_chk:
                return false;
            case llvm::LibFunc_abs:
                return true;
            case llvm::LibFunc_labs:
                return true;
            case llvm::LibFunc_llabs:
                return true;
            case llvm::LibFunc_access:
                return false;
            case llvm::LibFunc_acos:
                return true;
            case llvm::LibFunc_acosf:
                return true;
            case llvm::LibFunc_acosh:
                return true;
            case llvm::LibFunc_acoshf:
                return true;
            case llvm::LibFunc_acoshl:
                return true;
            case llvm::LibFunc_acosl:
                return true;
            case llvm::LibFunc_aligned_alloc:
                return false;
            case llvm::LibFunc_asin:
                return true;
            case llvm::LibFunc_asinf:
                return true;
            case llvm::LibFunc_asinh:
                return true;
            case llvm::LibFunc_asinhf:
                return true;
            case llvm::LibFunc_asinhl:
                return true;
            case llvm::LibFunc_asinl:
                return true;
            case llvm::LibFunc_atan:
                return true;
            case llvm::LibFunc_atan2:
                return true;
            case llvm::LibFunc_atan2f:
                return true;
            case llvm::LibFunc_atan2l:
                return true;
            case llvm::LibFunc_atanf:
                return true;
            case llvm::LibFunc_atanh:
                return true;
            case llvm::LibFunc_atanhf:
                return true;
            case llvm::LibFunc_atanhl:
                return true;
            case llvm::LibFunc_atanl:
                return true;
            case llvm::LibFunc_atof:
                return false;
            case llvm::LibFunc_atoi:
                return false;
            case llvm::LibFunc_atol:
                return false;
            case llvm::LibFunc_atoll:
                return false;
            case llvm::LibFunc_bcmp:
                return false;
            case llvm::LibFunc_bcopy:
                return false;
            case llvm::LibFunc_bzero:
                return false;
            case llvm::LibFunc_cabs:
                return false;
            case llvm::LibFunc_cabsf:
                return false;
            case llvm::LibFunc_cabsl:
                return false;
            case llvm::LibFunc_calloc:
                return true;
            case llvm::LibFunc_cbrt:
                return false;
            case llvm::LibFunc_cbrtf:
                return false;
            case llvm::LibFunc_cbrtl:
                return false;
            case llvm::LibFunc_ceil:
                return false;
            case llvm::LibFunc_ceilf:
                return false;
            case llvm::LibFunc_ceill:
                return false;
            case llvm::LibFunc_chmod:
                return false;
            case llvm::LibFunc_chown:
                return false;
            case llvm::LibFunc_clearerr:
                return false;
            case llvm::LibFunc_closedir:
                return false;
            case llvm::LibFunc_copysign:
                return false;
            case llvm::LibFunc_copysignf:
                return false;
            case llvm::LibFunc_copysignl:
                return false;
            case llvm::LibFunc_cos:
                return true;
            case llvm::LibFunc_cosf:
                return true;
            case llvm::LibFunc_cosh:
                return true;
            case llvm::LibFunc_coshf:
                return true;
            case llvm::LibFunc_coshl:
                return true;
            case llvm::LibFunc_cosl:
                return true;
            case llvm::LibFunc_ctermid:
                return false;
            case llvm::LibFunc_erf:
                return false;
            case llvm::LibFunc_erff:
                return false;
            case llvm::LibFunc_erfl:
                return false;
            case llvm::LibFunc_execl:
                return false;
            case llvm::LibFunc_execle:
                return false;
            case llvm::LibFunc_execlp:
                return false;
            case llvm::LibFunc_execv:
                return false;
            case llvm::LibFunc_execvP:
                return false;
            case llvm::LibFunc_execve:
                return false;
            case llvm::LibFunc_execvp:
                return false;
            case llvm::LibFunc_execvpe:
                return false;
            case llvm::LibFunc_exp:
                return true;
            case llvm::LibFunc_exp10:
                return true;
            case llvm::LibFunc_exp10f:
                return true;
            case llvm::LibFunc_exp10l:
                return true;
            case llvm::LibFunc_exp2:
                return true;
            case llvm::LibFunc_exp2f:
                return true;
            case llvm::LibFunc_exp2l:
                return true;
            case llvm::LibFunc_expf:
                return true;
            case llvm::LibFunc_expl:
                return true;
            case llvm::LibFunc_expm1:
                return true;
            case llvm::LibFunc_expm1f:
                return true;
            case llvm::LibFunc_expm1l:
                return true;
            case llvm::LibFunc_fabs:
                return true;
            case llvm::LibFunc_fabsf:
                return true;
            case llvm::LibFunc_fabsl:
                return true;
            case llvm::LibFunc_fclose:
                return false;
            case llvm::LibFunc_fdopen:
                return false;
            case llvm::LibFunc_feof:
                return false;
            case llvm::LibFunc_ferror:
                return false;
            case llvm::LibFunc_fflush:
                return false;
            case llvm::LibFunc_ffs:
                return false;
            case llvm::LibFunc_ffsl:
                return false;
            case llvm::LibFunc_ffsll:
                return false;
            case llvm::LibFunc_fgetc:
                return false;
            case llvm::LibFunc_fgetc_unlocked:
                return false;
            case llvm::LibFunc_fgetpos:
                return false;
            case llvm::LibFunc_fgets:
                return false;
            case llvm::LibFunc_fgets_unlocked:
                return false;
            case llvm::LibFunc_fileno:
                return false;
            case llvm::LibFunc_fiprintf:
                return false;
            case llvm::LibFunc_flockfile:
                return false;
            case llvm::LibFunc_floor:
                return true;
            case llvm::LibFunc_floorf:
                return true;
            case llvm::LibFunc_floorl:
                return true;
            case llvm::LibFunc_fls:
                return false;
            case llvm::LibFunc_flsl:
                return false;
            case llvm::LibFunc_flsll:
                return false;
            case llvm::LibFunc_fmax:
                return true;
            case llvm::LibFunc_fmaxf:
                return true;
            case llvm::LibFunc_fmaxl:
                return true;
            case llvm::LibFunc_fmin:
                return true;
            case llvm::LibFunc_fminf:
                return true;
            case llvm::LibFunc_fminl:
                return true;
            case llvm::LibFunc_fmod:
                return true;
            case llvm::LibFunc_fmodf:
                return true;
            case llvm::LibFunc_fmodl:
                return true;
            case llvm::LibFunc_fopen:
                return false;
            case llvm::LibFunc_fopen64:
                return false;
            case llvm::LibFunc_fork:
                return false;
            case llvm::LibFunc_fprintf:
                return false;
            case llvm::LibFunc_fputc:
                return false;
            case llvm::LibFunc_fputc_unlocked:
                return false;
            case llvm::LibFunc_fputs:
                return false;
            case llvm::LibFunc_fputs_unlocked:
                return false;
            case llvm::LibFunc_fread:
                return false;
            case llvm::LibFunc_fread_unlocked:
                return false;
            case llvm::LibFunc_free:
                return true;
            case llvm::LibFunc_frexp:
                return false;
            case llvm::LibFunc_frexpf:
                return false;
            case llvm::LibFunc_frexpl:
                return false;
            case llvm::LibFunc_fscanf:
                return false;
            case llvm::LibFunc_fseek:
                return false;
            case llvm::LibFunc_fseeko:
                return false;
            case llvm::LibFunc_fseeko64:
                return false;
            case llvm::LibFunc_fsetpos:
                return false;
            case llvm::LibFunc_fstat:
                return false;
            case llvm::LibFunc_fstat64:
                return false;
            case llvm::LibFunc_fstatvfs:
                return false;
            case llvm::LibFunc_fstatvfs64:
                return false;
            case llvm::LibFunc_ftell:
                return false;
            case llvm::LibFunc_ftello:
                return false;
            case llvm::LibFunc_ftello64:
                return false;
            case llvm::LibFunc_ftrylockfile:
                return false;
            case llvm::LibFunc_funlockfile:
                return false;
            case llvm::LibFunc_fwrite:
                return false;
            case llvm::LibFunc_fwrite_unlocked:
                return false;
            case llvm::LibFunc_getc:
                return false;
            case llvm::LibFunc_getc_unlocked:
                return false;
            case llvm::LibFunc_getchar:
                return false;
            case llvm::LibFunc_getchar_unlocked:
                return false;
            case llvm::LibFunc_getenv:
                return false;
            case llvm::LibFunc_getitimer:
                return false;
            case llvm::LibFunc_getlogin_r:
                return false;
            case llvm::LibFunc_getpwnam:
                return false;
            case llvm::LibFunc_gets:
                return false;
            case llvm::LibFunc_gettimeofday:
                return false;
            case llvm::LibFunc_htonl:
                return false;
            case llvm::LibFunc_htons:
                return false;
            case llvm::LibFunc_iprintf:
                return false;
            case llvm::LibFunc_isascii:
                return false;
            case llvm::LibFunc_isdigit:
                return false;
            case llvm::LibFunc_lchown:
                return false;
            case llvm::LibFunc_ldexp:
                return true;
            case llvm::LibFunc_ldexpf:
                return true;
            case llvm::LibFunc_ldexpl:
                return true;
            case llvm::LibFunc_log:
                return true;
            case llvm::LibFunc_log10:
                return true;
            case llvm::LibFunc_log10f:
                return true;
            case llvm::LibFunc_log10l:
                return true;
            case llvm::LibFunc_log1p:
                return true;
            case llvm::LibFunc_log1pf:
                return true;
            case llvm::LibFunc_log1pl:
                return true;
            case llvm::LibFunc_log2:
                return true;
            case llvm::LibFunc_log2f:
                return true;
            case llvm::LibFunc_log2l:
                return true;
            case llvm::LibFunc_logb:
                return true;
            case llvm::LibFunc_logbf:
                return true;
            case llvm::LibFunc_logbl:
                return true;
            case llvm::LibFunc_logf:
                return true;
            case llvm::LibFunc_logl:
                return true;
            case llvm::LibFunc_lstat:
                return false;
            case llvm::LibFunc_lstat64:
                return false;
            case llvm::LibFunc_malloc:
                return true;
            case llvm::LibFunc_memalign:
                return false;
            case llvm::LibFunc_memccpy:
                return false;
            case llvm::LibFunc_memchr:
                return false;
            case llvm::LibFunc_memcmp:
                return false;
            case llvm::LibFunc_memcpy:
                return false;
            case llvm::LibFunc_memmove:
                return false;
            case llvm::LibFunc_mempcpy:
                return false;
            case llvm::LibFunc_memrchr:
                return false;
            case llvm::LibFunc_memset:
                return false;
            case llvm::LibFunc_memset_pattern16:
                return false;
            case llvm::LibFunc_memset_pattern4:
                return false;
            case llvm::LibFunc_memset_pattern8:
                return false;
            case llvm::LibFunc_mkdir:
                return false;
            case llvm::LibFunc_mktime:
                return false;
            case llvm::LibFunc_modf:
                return false;
            case llvm::LibFunc_modff:
                return false;
            case llvm::LibFunc_modfl:
                return false;
            case llvm::LibFunc_nearbyint:
                return true;
            case llvm::LibFunc_nearbyintf:
                return true;
            case llvm::LibFunc_nearbyintl:
                return true;
            case llvm::LibFunc_ntohl:
                return false;
            case llvm::LibFunc_ntohs:
                return false;
            case llvm::LibFunc_open:
                return false;
            case llvm::LibFunc_open64:
                return false;
            case llvm::LibFunc_opendir:
                return false;
            case llvm::LibFunc_pclose:
                return false;
            case llvm::LibFunc_perror:
                return false;
            case llvm::LibFunc_popen:
                return false;
            case llvm::LibFunc_posix_memalign:
                return false;
            case llvm::LibFunc_pow:
                return true;
            case llvm::LibFunc_powf:
                return true;
            case llvm::LibFunc_powl:
                return true;
            case llvm::LibFunc_pread:
                return false;
            case llvm::LibFunc_printf:
                return false;
            case llvm::LibFunc_putc:
                return false;
            case llvm::LibFunc_putc_unlocked:
                return false;
            case llvm::LibFunc_putchar:
                return false;
            case llvm::LibFunc_putchar_unlocked:
                return false;
            case llvm::LibFunc_puts:
                return false;
            case llvm::LibFunc_pwrite:
                return false;
            case llvm::LibFunc_qsort:
                return false;
            case llvm::LibFunc_read:
                return false;
            case llvm::LibFunc_readlink:
                return false;
            case llvm::LibFunc_realloc:
                return false;
            case llvm::LibFunc_reallocf:
                return false;
            case llvm::LibFunc_realpath:
                return false;
            case llvm::LibFunc_remainder:
                return false;
            case llvm::LibFunc_remainderf:
                return false;
            case llvm::LibFunc_remainderl:
                return false;
            case llvm::LibFunc_remquo:
                return false;
            case llvm::LibFunc_remquof:
                return false;
            case llvm::LibFunc_remquol:
                return false;
            case llvm::LibFunc_remove:
                return false;
            case llvm::LibFunc_rename:
                return false;
            case llvm::LibFunc_rewind:
                return false;
            case llvm::LibFunc_rint:
                return true;
            case llvm::LibFunc_rintf:
                return true;
            case llvm::LibFunc_rintl:
                return true;
            case llvm::LibFunc_rmdir:
                return false;
            case llvm::LibFunc_round:
                return true;
            case llvm::LibFunc_roundeven:
                return true;
            case llvm::LibFunc_roundevenf:
                return true;
            case llvm::LibFunc_roundevenl:
                return true;
            case llvm::LibFunc_roundf:
                return true;
            case llvm::LibFunc_roundl:
                return true;
            case llvm::LibFunc_scanf:
                return false;
            case llvm::LibFunc_setbuf:
                return false;
            case llvm::LibFunc_setitimer:
                return false;
            case llvm::LibFunc_setvbuf:
                return false;
            case llvm::LibFunc_sin:
                return true;
            case llvm::LibFunc_sinf:
                return true;
            case llvm::LibFunc_sinh:
                return true;
            case llvm::LibFunc_sinhf:
                return true;
            case llvm::LibFunc_sinhl:
                return true;
            case llvm::LibFunc_sinl:
                return true;
            case llvm::LibFunc_siprintf:
                return false;
            case llvm::LibFunc_snprintf:
                return false;
            case llvm::LibFunc_sprintf:
                return false;
            case llvm::LibFunc_sqrt:
                return true;
            case llvm::LibFunc_sqrtf:
                return true;
            case llvm::LibFunc_sqrtl:
                return true;
            case llvm::LibFunc_sscanf:
                return false;
            case llvm::LibFunc_stat:
                return false;
            case llvm::LibFunc_stat64:
                return false;
            case llvm::LibFunc_statvfs:
                return false;
            case llvm::LibFunc_statvfs64:
                return false;
            case llvm::LibFunc_stpcpy:
                return false;
            case llvm::LibFunc_stpncpy:
                return false;
            case llvm::LibFunc_strcasecmp:
                return false;
            case llvm::LibFunc_strcat:
                return false;
            case llvm::LibFunc_strchr:
                return false;
            case llvm::LibFunc_strcmp:
                return false;
            case llvm::LibFunc_strcoll:
                return false;
            case llvm::LibFunc_strcpy:
                return false;
            case llvm::LibFunc_strcspn:
                return false;
            case llvm::LibFunc_strdup:
                return false;
            case llvm::LibFunc_strlcat:
                return false;
            case llvm::LibFunc_strlcpy:
                return false;
            case llvm::LibFunc_strlen:
                return false;
            case llvm::LibFunc_strncasecmp:
                return false;
            case llvm::LibFunc_strncat:
                return false;
            case llvm::LibFunc_strncmp:
                return false;
            case llvm::LibFunc_strncpy:
                return false;
            case llvm::LibFunc_strndup:
                return false;
            case llvm::LibFunc_strnlen:
                return false;
            case llvm::LibFunc_strpbrk:
                return false;
            case llvm::LibFunc_strrchr:
                return false;
            case llvm::LibFunc_strspn:
                return false;
            case llvm::LibFunc_strstr:
                return false;
            case llvm::LibFunc_strtod:
                return false;
            case llvm::LibFunc_strtof:
                return false;
            case llvm::LibFunc_strtok:
                return false;
            case llvm::LibFunc_strtok_r:
                return false;
            case llvm::LibFunc_strtol:
                return false;
            case llvm::LibFunc_strtold:
                return false;
            case llvm::LibFunc_strtoll:
                return false;
            case llvm::LibFunc_strtoul:
                return false;
            case llvm::LibFunc_strtoull:
                return false;
            case llvm::LibFunc_strxfrm:
                return false;
            case llvm::LibFunc_system:
                return false;
            case llvm::LibFunc_tan:
                return true;
            case llvm::LibFunc_tanf:
                return true;
            case llvm::LibFunc_tanh:
                return true;
            case llvm::LibFunc_tanhf:
                return true;
            case llvm::LibFunc_tanhl:
                return true;
            case llvm::LibFunc_tanl:
                return true;
            case llvm::LibFunc_times:
                return false;
            case llvm::LibFunc_tmpfile:
                return false;
            case llvm::LibFunc_tmpfile64:
                return false;
            case llvm::LibFunc_toascii:
                return false;
            case llvm::LibFunc_trunc:
                return false;
            case llvm::LibFunc_truncf:
                return false;
            case llvm::LibFunc_truncl:
                return false;
            case llvm::LibFunc_uname:
                return false;
            case llvm::LibFunc_ungetc:
                return false;
            case llvm::LibFunc_unlink:
                return false;
            case llvm::LibFunc_unsetenv:
                return false;
            case llvm::LibFunc_utime:
                return false;
            case llvm::LibFunc_utimes:
                return false;
            case llvm::LibFunc_valloc:
                return false;
            case llvm::LibFunc_vec_calloc:
                return false;
            case llvm::LibFunc_vec_free:
                return false;
            case llvm::LibFunc_vec_malloc:
                return false;
            case llvm::LibFunc_vec_realloc:
                return false;
            case llvm::LibFunc_vfprintf:
                return false;
            case llvm::LibFunc_vfscanf:
                return false;
            case llvm::LibFunc_vprintf:
                return false;
            case llvm::LibFunc_vscanf:
                return false;
            case llvm::LibFunc_vsnprintf:
                return false;
            case llvm::LibFunc_vsprintf:
                return false;
            case llvm::LibFunc_vsscanf:
                return false;
            case llvm::LibFunc_wcslen:
                return false;
            case llvm::LibFunc_write:
                return false;
            case llvm::NumLibFuncs:
                return false;
            case llvm::NotLibFunc:
                return false;
        }
    };
};

} // namespace lifting
} // namespace docc
