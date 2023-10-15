// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team
#include "../cppdebug.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cassert>
#include <vector>
#include "quantization.h"

template <typename T>
at::Tensor ds_quantize(at::Tensor& vals, int groups, int bits)
{
    debuginfo();
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if ((((size / groups) - 1) / 4096 + 1) <= 256) {
        launch_fake_quantize_kernel(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_sr_quantize(at::Tensor& vals, int groups, int bits)
{
    debuginfo();
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if (((size / groups) / 4 / 1024) <= 256) {
        launch_sr_fake_quantize_kernel(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_quantize_asym(at::Tensor& vals, int groups, int bits)
{
    debuginfo();
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if ((((size / groups) - 1) / 4096 + 1) <= 256) {
        launch_fake_quantize_kernel_asym(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

template <typename T>
at::Tensor ds_sr_quantize_asym(at::Tensor& vals, int groups, int bits)
{
    debuginfo();
    auto t_size = vals.sizes();
    int size = 1;
    for (auto dim : t_size) size *= dim;

    if (((size / groups) / 4 / 1024) <= 256) {
        launch_sr_fake_quantize_kernel_asym(
            (T*)vals.data_ptr(), size, groups, bits, at::cuda::getCurrentCUDAStream());
    }
    return vals;
}

std::vector<at::Tensor> quantize_kernel(at::Tensor& input_vals,
                                        int groups,
                                        int numBits,
                                        quantize::Type quantType)
{
    debuginfo();
    auto dtype = at::kFloat;
    auto params_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);
    const int param_elems = (quantize::requires_offset(quantType)) ? 2 : 1;
    auto params = torch::empty({groups, param_elems}, params_options);

    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    auto output_sizes = input_vals.sizes().vec();
    output_sizes[output_sizes.size() - 1] /= numBits == 8 ? 1 : 2;
    auto output = torch::empty(output_sizes, output_options);

    const int elems_per_group = at::numel(input_vals) / groups;

    launch_quant((int8_t*)output.data_ptr(),
                 (float*)params.data_ptr(),
                 (__half*)input_vals.data_ptr(),
                 groups,
                 elems_per_group,
                 numBits,
                 quantType,
                 at::cuda::getCurrentCUDAStream());

    return {output, params};
}

template <typename T>
at::Tensor dequantize(at::Tensor& quantized_data,
                      at::Tensor& params,
                      int groups,
                      int num_bits,
                      quantize::Type quant_type)
{
    debuginfo();
    auto dtype = (std::is_same<T, float>::value) ? torch::kFloat32 : torch::kFloat16;
    auto output_options = at::TensorOptions()
                              .dtype(dtype)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    auto output_sizes = quantized_data.sizes().vec();
    output_sizes[output_sizes.size() - 1] *= num_bits == 8 ? 1 : 2;
    auto output = torch::empty(output_sizes, output_options);

    const int total_elems = at::numel(output);
    const int elems_per_group = total_elems / groups;

    launch_dequantize_kernel((T*)output.data_ptr(),
                             (const int8_t*)quantized_data.data_ptr(),
                             (const float*)params.data_ptr(),
                             quant_type,
                             num_bits,
                             elems_per_group,
                             total_elems,
                             at::cuda::getCurrentCUDAStream());

    return output;
}

// 这个函数用于将输入数据集进行分组并进行量化处理。
std::vector<at::Tensor> ds_swizzle_quant(at::Tensor& input_vals,
                                         int groups,
                                         int num_bits,
                                         quantize::Type quant_type,
                                         int pipeline_size,
                                         int nodes,
                                         int devices_per_node)
{
    debuginfo();
    // 定义了一个at::TensorOptions对象，它描述了接下来要创建的张量的属性。
    // 这个张量的数据类型是float，布局是strided，设备是CUDA设备，且不需要计算梯度。
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    // 通过检查量化类型是否需要偏移，来确定比例因子的数量。
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;

    // 创建一个未初始化的张量，其大小为{groups, scales_elems}，并使用之前定义的张量属性。
    auto scales = torch::empty({groups, scales_elems}, scales_options);

    // 同样地，创建了一个未初始化的张量用于存储输出结果。其数据类型是char，
    // 布局是strided，设备是CUDA设备，且不需要计算梯度。
    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    // 计算量化因子，它由8位除以量化位数得出。
    const int quantization_scalar = 8 / num_bits;

    // 计算量化后的值的数量，通过输入值的元素总数除以量化因子得出。
    const int compressed_vals = at::numel(input_vals) / quantization_scalar;

    // 创建一个未初始化的张量，用于存储量化后的值。
    auto output = torch::empty({compressed_vals}, output_options);

    // 计算每组的元素数量，通过输入值的元素总数除以组数得出。
    const int elems_per_group = at::numel(input_vals) / groups;

    // 调用之前定义的函数launch_swizzled_quant，对输入的张量进行量化操作。
    // 参数包括输入的数据、量化位数、量化类型、组数、每组的元素数量等等。
    launch_swizzled_quant((int8_t*)output.data_ptr(),
                          (float*)scales.data_ptr(),
                          (__half*)input_vals.data_ptr(),
                          num_bits,
                          quant_type,
                          groups,
                          elems_per_group,
                          pipeline_size,
                          nodes,
                          devices_per_node,
                          at::cuda::getCurrentCUDAStream());

    // 返回一个包含两个元素的向量，第一个元素是量化后的值，第二个元素是量化的缩放因子。
    return {output, scales};
}

// 这是一个将输入的量化数据进行降维和反量化的操作
std::vector<at::Tensor> quantized_reduction(at::Tensor& input_vals,
                                            at::Tensor& input_scales,
                                            int in_groups,
                                            int out_groups,
                                            int num_bits,
                                            quantize::Type quant_type)
{
    debuginfo();
    // 定义一个TensorOptions对象scales_options，表示接下来要创建的张量的属性，
    // 这个张量的数据类型是float，布局是strided，设备是CUDA设备，并且不需要计算梯度。
    auto scales_options = at::TensorOptions()
                              .dtype(at::kFloat)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    // 根据量化类型是否需要偏移量，确定量化缩放因子的数量。
    const int scales_elems = (quantize::requires_offset(quant_type)) ? 2 : 1;

    // 使用scales_options定义一个空的张量scales，大小为{out_groups, scales_elems}，用来存储量化缩放因子。
    auto scales = torch::empty({out_groups, scales_elems}, scales_options);

    // 定义一个新的TensorOptions对象output_options，表示接下来要创建的输出张量的属性，
    // 这个张量的数据类型是char，布局是strided，设备是CUDA设备，并且不需要计算梯度。
    auto output_options = at::TensorOptions()
                              .dtype(at::kChar)
                              .layout(at::kStrided)
                              .device(at::kCUDA)
                              .requires_grad(false);

    // 将input_vals的大小转化为一个std::vector<long int>对象。
    std::vector<long int> sz(input_vals.sizes().begin(), input_vals.sizes().end());

    // 这里假设每个节点上有16个GPU。这个值可能会根据实际的机器配置有所不同。
    const int gpu_per_node = 16;                   // depend on machine in_groups/out_groups;

    // 修改最后一个维度的大小，使其等于原来的大小除以节点上的GPU数量。这可能是为了将数据在节点的各个GPU之间进行分割。
    sz[sz.size() - 1] = sz.back() / gpu_per_node;  // num of GPU per nodes

    // 计算每个GPU处理的输入元素数量。
    const int elems_per_in_tensor = at::numel(input_vals) / gpu_per_node;

    // 创建一个空的张量output，其大小为sz，用于存储输出结果。
    auto output = torch::empty(sz, output_options);

    // 计算每个输入组和每个输出组的元素数量。
    const int elems_per_in_group = elems_per_in_tensor / (in_groups / gpu_per_node);
    const int elems_per_out_group = elems_per_in_tensor / out_groups;

    // 调用之前定义的launch_dequant_reduce函数，对输入的张量进行降维和反量化操作。
    // 参数包括输出张量、输入张量、量化比例、GPU数量、量化位数、量化类型、输出组数、
    // 每个输出组的元素数量、每个输入张量的元素数量、每个GPU处理的输入组数、每个输入组的元素数量等。
    launch_dequant_reduce((int8_t*)output.data_ptr(),
                          (float*)scales.data_ptr(),
                          (const int8_t*)input_vals.data_ptr(),
                          (const float*)input_scales.data_ptr(),
                          gpu_per_node,
                          num_bits,
                          quant_type,
                          out_groups,
                          elems_per_out_group,
                          elems_per_in_tensor,
                          in_groups / gpu_per_node,
                          elems_per_in_group,
                          at::cuda::getCurrentCUDAStream());

    // 返回一个包含两个元素的向量，第一个元素是输出结果，第二个元素是量化缩放因子。
    return {output, scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ds_quantize_fp32", &ds_quantize<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_quantize_fp16", &ds_quantize<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_sr_quantize_fp32", &ds_sr_quantize<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_sr_quantize_fp16", &ds_sr_quantize<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_quantize_asym_fp32", &ds_quantize_asym<float>, "DeepSpeed Quantize with fp32 (CUDA)");
    m.def(
        "ds_quantize_asym_fp16", &ds_quantize_asym<__half>, "DeepSpeed Quantize with fp16 (CUDA)");
    m.def("ds_sr_quantize_asym_fp32",
          &ds_sr_quantize_asym<float>,
          "DeepSpeed Quantize with fp32 (CUDA)");
    m.def("ds_sr_quantize_asym_fp16",
          &ds_sr_quantize_asym<__half>,
          "DeepSpeed Quantize with fp16 (CUDA)");
    pybind11::enum_<quantize::Type>(m, "QuantizationType")
        .value("Symmetric", quantize::Type::Symmetric)
        .value("Asymmetric", quantize::Type::Asymmetric)
        .export_values();
    m.def("quantize", &quantize_kernel);
    m.def("dequantize", &dequantize<__half>);
    m.def("dequantize_fp32", &dequantize<float>);
    m.def("swizzle_quant", &ds_swizzle_quant);
    m.def("quantized_reduction", &quantized_reduction);
}
