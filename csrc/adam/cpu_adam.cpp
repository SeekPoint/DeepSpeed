// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
#include "../cppdebug.h"
#include "cpu_adam.h"
#include <torch/extension.h>
#include <cassert>
#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>

#if defined(__ENABLE_CUDA__)
#include <cuda_runtime_api.h>
#include "cublas_v2.h"
#include "cuda.h"
#include "curand.h"
#include "custom_cuda_layers.h"
#endif

// 用于保存优化器实例的全局变量
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

// Adam优化器的一步操作，处理一次优化步骤
// 具体的优化步骤，包括计算梯度、更新参数等
void Adam_Optimizer::Step_1(float *_params,                  // 参数数组
                            float *grads,                    // 梯度数组
                            float *_exp_avg,                 // 梯度的指数移动平均值
                            float *_exp_avg_sq,              // 梯度平方的指数移动平均值
                            size_t _param_size,              // 参数数组的大小
                            ds_half_precision_t *dev_params, // 设备参数
                            bool half_precision)             // 是否使用半精度
{
    debuginfo();
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    // 如果定义了AVX512或AVX256，使用AVX版本的步骤
    Step_AVX<1>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size) {
        // 计算一些常量
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;
        if (half_precision) {
            // 如果使用半精度，进行类型转换
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        // 遍历未处理的部分
        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;

#if defined(__ENABLE_CUDA__)
            // 如果是在GPU上运行，需要同步CUDA流
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#endif
#pragma omp parallel for

            // 对每个元素进行更新
            for (size_t k = t; k < offset; k++) {
                // 读取参数和对应的梯度
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];

                // 根据模式选择是否添加权重衰减
                if (_weight_decay > 0 && !_adamw_mode) { grad = param * _weight_decay + grad; }

                // 更新指数移动平均值
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                // 计算新的梯度值
                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;

                // 根据模式选择是否添加权重衰减
                if (_weight_decay > 0 && _adamw_mode) { param += w_decay * param; }

                // 更新参数值
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__)
                // 如果是在GPU上运行，更新设备缓冲区
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                // 保存新的参数值
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;

                // 保存新的指数移动平均值
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
#if defined(__ENABLE_CUDA__)
            // 如果是在GPU上运行，更新设备参数
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);

                _buf_index = !_buf_index;
            }
#endif
        }
    }
}

// 处理4次优化步骤，如果参数数量大于已处理的参数数量，调用Step_1来处理剩余的参数
// 处理4次优化步骤
// 如果还有剩余参数，调用Step_1处理剩余参数
void Adam_Optimizer::Step_4(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    debuginfo();
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<4>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

// 创建Adam优化器的函数声明
// 创建一个新的Adam优化器，并将它存储在全局变量s_optimizers中
int create_adam_optimizer(int optimizer_id,
                          float alpha = 1e-3,
                          float betta1 = 0.9,
                          float betta2 = 0.999,
                          float eps = 1e-8,
                          float weight_decay = 0,
                          bool adamw_mode = true,
                          bool should_log = false)
{
    debuginfo();

    // 创建一个新的Adam优化器
    auto opt =
        std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    // 将新创建的优化器添加到全局变量中
    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
#if defined(__AVX512__)
        avx_type = "AVX512";
#else
#if defined(__AVX256__)
        avx_type = "AVX2";
#else
        avx_type = "scalar";
#endif
#endif
        // 如果需要打印日志，打印创建的优化器信息
        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}


// 处理8次优化步骤，如果参数数量大于已处理的参数数量，调用Step_4来处理剩余的参数
// 处理8次优化步骤
// 如果还有剩余参数，调用Step_4处理剩余参数
void Adam_Optimizer::Step_8(float* _params,
                            float* grads,
                            float* _exp_avg,
                            float* _exp_avg_sq,
                            size_t _param_size,
                            ds_half_precision_t* dev_params,
                            bool half_precision)
{
    debuginfo();
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    Step_AVX<8>(&rounded_size,
                _params,
                grads,
                _exp_avg,
                _exp_avg_sq,
                _param_size,
                dev_params,
                half_precision);
#endif
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

// Adam优化器步骤的函数声明
// 使用指定的优化器进行一步优化操作
int ds_adam_step(int optimizer_id,
                 size_t step,
                 float lr,
                 float beta1,
                 float beta2,
                 float epsilon,
                 float weight_decay,
                 bool bias_correction,
                 torch::Tensor& params,
                 torch::Tensor& grads,
                 torch::Tensor& exp_avg,
                 torch::Tensor& exp_avg_sq)
{
    debuginfo();
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    // 从全局变量中获取指定的优化器
    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    // 调用优化器的函数进行一步优化
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                nullptr,
                (params.options().dtype() == at::kHalf));

#if defined(__ENABLE_CUDA__)
    opt->SynchronizeStreams();
#endif
    return 0;
}

// 带有参数复制的Adam优化器步骤的函数声明
// 使用指定的优化器进行一步优化，并将结果复制到GPU中
int ds_adam_step_plus_copy(int optimizer_id,
                           size_t step,
                           float lr,
                           float beta1,
                           float beta2,
                           float epsilon,
                           float weight_decay,
                           bool bias_correction,
                           torch::Tensor& params,
                           torch::Tensor& grads,
                           torch::Tensor& exp_avg,
                           torch::Tensor& exp_avg_sq,
                           torch::Tensor& gpu_params)
{
    debuginfo();
#if defined(__ENABLE_CUDA__)
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    ds_half_precision_t* gpu_params_ptr = (ds_half_precision_t*)gpu_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    // 从全局变量中获取指定的优化器
    std::shared_ptr<Adam_Optimizer> opt =
        std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);

    // 调用优化器的函数进行一步优化，并将结果复制到GPU中
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                gpu_params_ptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
#else
    assert(false);
#endif
    return 0;
}

// 销毁Adam优化器的函数声明
// 销毁指定的优化器
int destroy_adam_optimizer(int optimizer_id)
{
    debuginfo();
    // 从全局变量s_optimizers中移除指定的优化器
    s_optimizers.erase(optimizer_id);

    return 0;
}

// 使用 pybind11 定义一个 Python 模块，该模块是一个 torch 扩展
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // 向模块中添加一个 Python 函数 'adam_update'。此函数绑定到 C++ 函数 'ds_adam_step'
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");

    // 向模块中添加一个 Python 函数 'adam_update_copy'。此函数绑定到 C++ 函数 'ds_adam_step_plus_copy'
    m.def("adam_update_copy",
          &ds_adam_step_plus_copy,
          "DeepSpeed CPU Adam update and param copy (C++)");

    // 向模块中添加一个 Python 函数 'create_adam'。此函数绑定到 C++ 函数 'create_adam_optimizer'
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");

    // 向模块中添加一个 Python 函数 'destroy_adam'。此函数绑定到 C++ 函数 'destroy_adam_optimizer'
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
}
