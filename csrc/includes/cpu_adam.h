// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#pragma once

#define NOMINMAX  // Windows idiosyncrasy
                  // https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c
// Windows的特殊性质，详情可见（这个没看，可能是在window上运行会有问题，但是我不在window上运行，需要可自行关注）
// https://stackoverflow.com/questions/4913922/possible-problems-with-nominmax-on-visual-c

#include <stdio.h>
#include <cassert>
#include "simd.h"

#if defined(__ENABLE_CUDA__)  // 如果定义了__ENABLE_CUDA__，则包含CUDA相关的头文件
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include "cuda.h"
#include "custom_cuda_layers.h"
typedef __half ds_half_precision_t;   // 定义半精度浮点类型
#else
#include <cmath>
// 如果没有定义__ENABLE_CUDA__，则使用unsigned short作为半精度浮点类型
typedef unsigned short ds_half_precision_t;  // 使用半精度浮点数，如果启用了CUDA
#endif

#define STEP(SPAN)                                             \
    void Step_##SPAN(float* _params,                           \
                     float* grads,                             \
                     float* _exp_avg,                          \
                     float* _exp_avg_sq,                       \
                     size_t _param_size,                       \
                     ds_half_precision_t* dev_param = nullptr, \
                     bool half_precision = false);
// 定义一个宏，用于创建不同步长的优化步骤函数， 于生成不同SPAN的Step函数

// 定义一个Adam优化器类
class Adam_Optimizer {
public:
    Adam_Optimizer(float alpha = 1e-3,
                   float betta1 = 0.9,
                   float betta2 = 0.999,
                   float eps = 1e-8,
                   float weight_decay = 0,
                   bool adamw_mode = true)
        : _alpha(alpha),
          _betta1(betta1),
          _betta2(betta2),
          _eps(eps),
          _weight_decay(weight_decay),
          _betta1_t(1.0),
          _betta2_t(1.0),
          _step(0),
          _adamw_mode(adamw_mode)
    {
// 如果定义了__ENABLE_CUDA__，则分配和初始化CUDA相关的资源
#if defined(__ENABLE_CUDA__)
        // 如果启用了CUDA，分配双缓冲区
        cudaMallocHost((void**)_doubled_buffer, TILE * sizeof(float));
        cudaMallocHost((void**)(_doubled_buffer + 1), TILE * sizeof(float));

        _streams[0] = TrainingContext::Instance().GetCurrentStream();  // 获取当前CUDA流
        _streams[1] = TrainingContext::Instance().GetNewStream();      // 获取新的CUDA流
        _buf_index = false;
#endif
    }
    ~Adam_Optimizer()  // 析构函数，用于清理资源 释放相关资源
    {
// 如果定义了__ENABLE_CUDA__，则释放CUDA相关的资源
#if defined(__ENABLE_CUDA__)
        cudaFreeHost(_doubled_buffer[0]);
        cudaFreeHost(_doubled_buffer[1]);
#endif
    }
// 如果定义了__AVX512__或__AVX256__，则生成AVX版本的Step函数
#if defined(__AVX512__) or defined(__AVX256__)
    template <int span>
    void Step_AVX(size_t* rounded_size,
                  float* _params,
                  float* grads,
                  float* _exp_avg,
                  float* _exp_avg_sq,
                  size_t param_size,
                  ds_half_precision_t* dev_param = nullptr,
                  bool half_precision = false);   // 如果启用了AVX512或AVX256，定义一个特化的步进函数
#endif
    STEP(1) // 定义步长为1的优化步骤函数
    STEP(4) // 定义步长为4的优化步骤函数
    STEP(8) // 定义步长为8的优化步骤函数
// 如果定义了__ENABLE_CUDA__，则生成用于同步CUDA流的函数
#if defined(__ENABLE_CUDA__)
    inline void SynchronizeStreams()  // CUDA流同步函数
    {
        for (int i = 0; i < 2; i++) cudaStreamSynchronize(_streams[i]);
    }
#endif
    // 用于更新步骤和beta值的函数
    // 增加步数
    inline void IncrementStep(size_t step, float beta1, float beta2)
    {
        // 如果传入的beta值与当前的beta值不匹配，更新beta值和步骤数，并计算相应的beta的幂次
        if (beta1 != _betta1 || beta2 != _betta2) {
            _step = step;                        // 更新步骤数
            _betta1 = beta1;                     // 更新beta1
            _betta2 = beta2;                     // 更新beta2
            _betta1_t = std::pow(_betta1, step); // 计算beta1的指数
            _betta2_t = std::pow(_betta2, step); // 计算beta2的指数
        } else {
            // 如果beta值未改变，则只更新步骤数，并根据新的步骤数更新beta的幂次
            _step++;
            if (_step != step) {
                _betta1_t = std::pow(_betta1, step);
                _betta2_t = std::pow(_betta2, step);
                _step = step;
            } else {
                _betta1_t *= _betta1;
                _betta2_t *= _betta2;
            }
        }
    }

    // 用于更新学习率、epsilon值、权重衰减和偏差矫正的函数
    // 更新状态
    inline void update_state(float lr, float epsilon, float weight_decay, bool bias_correction)
    {
        // 更新学习率、epsilon值和权重衰减
        _alpha = lr;                  // 更新学习率
        _eps = epsilon;               // 更新epsilon值
        _weight_decay = weight_decay; // 更新权重衰减

        // 初始化偏差矫正因子
        _bias_correction1 = 1.0f;
        _bias_correction2 = 1.0f;

        // 如果需要偏差矫正，更新偏差矫正因子
        if (bias_correction == 1) {
            _bias_correction1 = 1 - _betta1_t;            // 更新偏差矫正因子1
            _bias_correction2 = 1 / sqrt(1 - _betta2_t);  // 更新偏差矫正因子2
        }
    }

private:
    float _alpha;        // 学习率
    float _betta1;       // beta1 参数
    float _betta2;       // beta2 参数
    float _eps;          // epsilon 参数  用于防止除零错误
    float _weight_decay; // 权重衰减参数

    float _betta1_t; // beta1 的幂次
    float _betta2_t; // beta2 的幂次
    size_t _step;    // 步骤计数器

    float _bias_correction1; // 偏差矫正因子1
    float _bias_correction2; // 偏差矫正因子2

    bool _adamw_mode; // AdamW 模式标志

#if defined(__ENABLE_CUDA__)   // 如果定义了__ENABLE_CUDA__，则定义CUDA相关的成员变量
    // 如果启用了 CUDA，定义一些额外的私有变量
    float *_doubled_buffer[2]; // 双缓冲区
    cudaStream_t _streams[2];  // CUDA 流
    bool _buf_index;           // 缓冲区索引
    #endif
};

#if defined(__AVX512__) or defined(__AVX256__)
// 使用 AVX 的优化器步骤函数
template <int span>
void Adam_Optimizer::Step_AVX(size_t* rounded_size,
                              float* _params,
                              float* grads,
                              float* _exp_avg,
                              float* _exp_avg_sq,
                              size_t _param_size,
                              ds_half_precision_t* dev_params,
                              bool half_precision)
{
    // 初始化新的舍入大小
    size_t new_rounded_size = 0;

    // 判断是否为半精度
    int rshft = half_precision ? 1 : 0;

    // 设置 beta1 和 beta2 的 SIMD 数据
    AVX_Data betta1_4;
    betta1_4.data = SIMD_SET(_betta1);
    AVX_Data betta2_4;
    betta2_4.data = SIMD_SET(_betta2);

    // 计算 1 - beta1 和 1 - beta2
    float betta1_minus1 = 1 - _betta1;
    float betta2_minus1 = 1 - _betta2;

    // 设置 1 - beta1 和 1 - beta2 的 SIMD 数据
    AVX_Data betta1_minus1_4;
    betta1_minus1_4.data = SIMD_SET(betta1_minus1);
    AVX_Data betta2_minus1_4;
    betta2_minus1_4.data = SIMD_SET(betta2_minus1);

    // 设置偏差矫正因子2的 SIMD 数据
    AVX_Data bias2_sqrt;
    bias2_sqrt.data = SIMD_SET(_bias_correction2);

    // 设置 epsilon 的 SIMD 数据
    AVX_Data eps_4;
    eps_4.data = SIMD_SET(_eps);

    // 计算步长大小  // 计算步长，步长等于学习率的负数
    float step_size = -1 * _alpha / _bias_correction1;

    // 设置步长大小的 SIMD 数据 // 初始化权重衰减变量
    AVX_Data step_size_4;
    step_size_4.data = SIMD_SET(step_size);

    // 计算权重衰减
    float w_decay = -1 * _alpha * _weight_decay;
    AVX_Data weight_decay4;
    if (_weight_decay > 0)
        // 如果启用了 AdamW 模式，设置权重衰减的 SIMD 数据
        weight_decay4.data = (_adamw_mode ? SIMD_SET(w_decay) : SIMD_SET(_weight_decay));

    // 设置新的舍入大小
    new_rounded_size = ROUND_DOWN(_param_size, SIMD_WIDTH * span);

    // 主循环  // 对参数进行循环
    for (size_t t = 0; t < new_rounded_size; t += TILE) {
        size_t copy_size = TILE;
        if ((t + TILE) > new_rounded_size) copy_size = new_rounded_size - t;
        size_t offset = copy_size + t;
#if defined(__ENABLE_CUDA__)
        // 如果启用了 CUDA，同步 CUDA 流
        if ((t / TILE) >= 2) { cudaStreamSynchronize(_streams[_buf_index]); }
#endif
#pragma omp parallel for  // 使用OpenMP并行化for循环
        // 对每个参数进行并行处理
        for (size_t i = t; i < offset; i += SIMD_WIDTH * span) {
            // 在SIMD指令的帮助下并行处理数据
            // 加载参数及相关数据
            AVX_Data grad_4[span];                                         // 创建梯度数据的AVX数组
            simd_load<span>(grad_4, grads + (i >> rshft), half_precision); // 从内存中加载梯度数据

            AVX_Data momentum_4[span];                        // 创建动量数据的AVX数组
            simd_load<span>(momentum_4, _exp_avg + i, false); // 从内存中加载动量数据

            AVX_Data variance_4[span];                           // 创建方差数据的AVX数组
            simd_load<span>(variance_4, _exp_avg_sq + i, false); // 从内存中加载方差数据

            AVX_Data param_4[span];                                           // 创建参数数据的AVX数组
            simd_load<span>(param_4, _params + (i >> rshft), half_precision); // 从内存中加载参数数据

            // 如果存在权重衰减，则应用权重衰减
            if (_weight_decay > 0 && !_adamw_mode)
            {                                                           // 如果权重衰减大于0且不是AdamW模式
                simd_fma<span>(grad_4, param_4, weight_decay4, grad_4); // 计算梯度的权重衰减
            }

            // 更新动量和方差 // 计算梯度和参数更新
            simd_mul<span>(momentum_4, momentum_4, betta1_4);
            simd_fma<span>(momentum_4, grad_4, betta1_minus1_4, momentum_4);
            simd_mul<span>(variance_4, variance_4, betta2_4);
            simd_mul<span>(grad_4, grad_4, grad_4);
            simd_fma<span>(variance_4, grad_4, betta2_minus1_4, variance_4);

            // 计算新的梯度
            simd_sqrt<span>(grad_4, variance_4);
            simd_fma<span>(grad_4, grad_4, bias2_sqrt, eps_4);
            simd_div<span>(grad_4, momentum_4, grad_4);

            if (_weight_decay > 0 && _adamw_mode) {
                // 如果权重衰减大于0且是AdamW模式
                simd_fma<span>(param_4, param_4, weight_decay4, param_4); // 计算参数的权重衰减
            }

            // 更新参数
            simd_fma<span>(param_4, grad_4, step_size_4, param_4);

            // 将新的参数、动量和方差存回内存 // 存储更新后的参数
            simd_store<span>(_params + (i >> rshft), param_4, half_precision);
#if defined(__ENABLE_CUDA__)
            if (dev_params) {
                simd_store<span>(_doubled_buffer[_buf_index] + (i - t), param_4, half_precision);
            }
#endif
            simd_store<span>(_exp_avg + i, momentum_4, false);
            simd_store<span>(_exp_avg_sq + i, variance_4, false);
        }
#if defined(__ENABLE_CUDA__)
        if (dev_params) {
            // 如果启用了CUDA，根据精度将参数更新到GPU
            if (half_precision)
                launch_param_update_half(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);
            else
                launch_param_update(
                    _doubled_buffer[_buf_index], dev_params + t, copy_size, _streams[_buf_index]);

            _buf_index = !_buf_index;  // 切换缓冲区索引
        }
#endif
    }
    *rounded_size = new_rounded_size; // 更新rounded_size值
}
#endif
