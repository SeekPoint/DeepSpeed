// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
#include "../cppdebug.h"
#include "cpu_adagrad.h"
#include <torch/extension.h>
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

// 用于保存优化器实例的全局变量  // 保存优化器的哈希表
static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

// Adam优化器的一步操作，处理一次优化步骤  // 具体的优化步骤，包括计算梯度、更新参数等
void Adagrad_Optimizer::Step_1(float* _params,      // 参数
                               float* grads,        // 梯度
                               float* _exp_avg_sq,  // 平方梯度的滑动平均值
                               size_t _param_size,  // 参数的大小
                               ds_half_precision_t* dev_params,
                               bool half_precision)
{
    debuginfo();
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    // 如果定义了AVX，则使用AVX版本的步骤
    Step_AVX<1>(
        &rounded_size, _params, grads, _exp_avg_sq, _param_size, dev_params, half_precision);
#endif
    // 如果参数的大小大于rounded_size
    if (_param_size > rounded_size) {
        // 步长
        float step_size = -1 * _alpha;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;

        // 如果半精度为真
        if (half_precision) {
            // 对梯度和参数进行类型转换
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        // 从rounded_size开始，以TILE为步长，遍历参数
        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            // 如果 t+TILE 大于参数的大小，则 copy_size 等于参数的大小减 t
            if ((t + TILE) > _param_size) copy_size = _param_size - t;
            size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
            // 如果 t/TILE 大于等于2，则同步CUDA流
            if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#endif
            // 并行处理每个元素
#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                // 根据精度获取梯度和参数
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = grads[k];
                float variance = _exp_avg_sq[k];

                // 如果权重衰减大于0，则梯度等于参数乘以权重衰减加上梯度
                if (_weight_decay > 0) { grad = param * _weight_decay + grad; }

                // 平方梯度的滑动平均值加上梯度的平方
                variance += grad * grad;

                // 梯度等于平方梯度的滑动平均值的平方根
                grad = sqrt(variance);
                grad += _eps;
                grad = momentum / grad;
                param = grad * step_size + param;
#if defined(__ENABLE_CUDA__)
                // 如果 dev_params 不为空，则更新 _doubled_buffer
                if (dev_params) _doubled_buffer[_buf_index][k - t] = param;
#endif
                // 更新参数
                if (half_precision)
                    params_cast_h[k] = (ds_half_precision_t)param;
                else
                    _params[k] = param;

                // STORE UPDATE TERM TO GRAD'S MEMORY
                // 将更新项存储到梯度的内存中
                grads[k] = grad * step_size;

                // 更新平方梯度的滑动平均值
                _exp_avg_sq[k] = variance;
            }
#if defined(__ENABLE_CUDA__)
            // 如果 dev_params 不为空，启动参数更新
            if (dev_params) {
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, (copy_size), _streams[_buf_index]);
                // 更新缓冲区索引
                _buf_index = !_buf_index;
            }
#endif
        }
    }
}


// 处理4次优化步骤，如果参数数量大于已处理的参数数量，调用Step_1来处理剩余的参数
// Adagrad优化器的四步
// 如果还有剩余参数，调用Step_1处理剩余参数
void Adagrad_Optimizer::Step_4(float* _params,      // 参数
                               float* grads,        // 梯度
                               float* _exp_avg_sq,  // 平方梯度的滑动平均值
                               size_t _param_size,  // 参数的大小
                               ds_half_precision_t* dev_params,
                               bool half_precision)
{
    debuginfo();
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    // 如果定义了AVX，则使用AVX版本的步骤
    Step_AVX<4>(
        &rounded_size, _params, grads, _exp_avg_sq, _param_size, dev_params, half_precision);
#endif
    // 如果参数的大小大于rounded_size
    if (_param_size > rounded_size)
        // 执行第一步
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

// 创建Adagrad优化器
int create_adagrad_optimizer(int optimizer_id,
                             float alpha = 1e-2,
                             float eps = 1e-8,
                             float weight_decay = 0,
                             bool should_log = false)
{
    debuginfo();
    // 创建一个Adagrad优化器对象
    auto opt = std::make_shared<Adagrad_Optimizer>(alpha, eps, weight_decay);

    // 将新创建的优化器存储到全局的优化器管理器中
    s_optimizers[optimizer_id] = opt;

    // 如果需要打印日志
    if (should_log) {
        // 判断处理器的向量计算能力
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

        printf("Adagrad Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, weight_decay=%f\n", alpha, weight_decay);
    }

    // 返回0表示成功
    return 0;
}


// 处理8次优化步骤，如果参数数量大于已处理的参数数量，调用Step_4来处理剩余的参数
// // Adagrad优化器的更新步骤
// 如果还有剩余参数，调用Step_4处理剩余参数
void Adagrad_Optimizer::Step_8(float* _params,
                               float* grads,
                               float* _exp_avg_sq,
                               size_t _param_size,
                               ds_half_precision_t* dev_params,
                               bool half_precision)
{
    debuginfo();
    // 初始化向量化计算的大小
    size_t rounded_size = 0;
#if defined(__AVX512__) or defined(__AVX256__)
    // 如果支持AVX，进行向量化计算
    Step_AVX<8>(
        &rounded_size, _params, grads, _exp_avg_sq, _param_size, dev_params, half_precision);
#endif
    // 对剩余的部分进行计算
    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

// Adagrad优化器的更新步骤
int ds_adagrad_step(int optimizer_id,
                    size_t step,
                    float lr,
                    float epsilon,
                    float weight_decay,
                    torch::Tensor& params,
                    torch::Tensor& grads,
                    torch::Tensor& exp_avg_sq)
{
    debuginfo();
    // 获取连续的数据
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // 获取指针
    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    // 获取优化器
    std::shared_ptr<Adagrad_Optimizer> opt =
        std::static_pointer_cast<Adagrad_Optimizer>(s_optimizers[optimizer_id]);

    // 更新步数
    opt->IncrementStep(step);

    // 更新状态
    opt->update_state(lr, epsilon, weight_decay);

    // 执行更新步骤
    opt->Step_8(params_ptr, grads_ptr, exp_avg_sq_ptr, params_c.numel());

#if defined(__ENABLE_CUDA__)
    // 同步CUDA流
    opt->SynchronizeStreams();
#endif
    // 返回0表示成功
    return 0;
}

// Adagrad优化器的更新步骤，并复制参数到GPU
int ds_adagrad_step_plus_copy(int optimizer_id,
                              size_t step,
                              float lr,
                              float epsilon,
                              float weight_decay,
                              torch::Tensor& params,
                              torch::Tensor& grads,
                              torch::Tensor& exp_avg_sq,
                              torch::Tensor& gpu_params)
{
    debuginfo();
#if defined(__ENABLE_CUDA__)
    // 获取连续的数据
    auto params_c = params.contiguous();
    auto gpu_params_c = gpu_params.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    // 获取指针
    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    ds_half_precision_t* gpu_params_ptr = (ds_half_precision_t*)gpu_params_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    // 获取优化器
    std::shared_ptr<Adagrad_Optimizer> opt =
        std::static_pointer_cast<Adagrad_Optimizer>(s_optimizers[optimizer_id]);

    // 更新步数
    opt->IncrementStep(step);

    // 更新状态
    opt->update_state(lr, epsilon, weight_decay);

    // 执行更新步骤，并复制参数到GPU
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_sq_ptr,
                params_c.numel(),
                gpu_params_ptr,
                (params.options().dtype() == at::kHalf));

    // 同步CUDA流
    opt->SynchronizeStreams();
#else
    // 如果没有启用CUDA，直接报错退出
    assert(false);
#endif
    // 返回0表示成功
    return 0;
}

// 销毁Adagrad优化器
int destroy_adagrad_optimizer(int optimizer_id)
{
    // 从全局的优化器管理器中移除
    s_optimizers.erase(optimizer_id);
    // 返回0表示成功
    return 0;
}

// Python绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adagrad_update", &ds_adagrad_step, "DeepSpeed CPU Adagrad update (C++)");
    m.def("adagrad_update_copy",
          &ds_adagrad_step_plus_copy,
          "DeepSpeed CPU Adagrad update and param copy (C++)");
    m.def("create_adagrad", &create_adagrad_optimizer, "DeepSpeed CPU Adagrad (C++)");
    m.def("destroy_adagrad", &destroy_adagrad_optimizer, "DeepSpeed CPU Adagrad destroy (C++)");
}
