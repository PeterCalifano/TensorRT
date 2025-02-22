/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "logger.h"
#include "util.h"
#include <cuda_runtime_api.h>

constexpr long long operator"" _MiB(long long unsigned val)
{
    return val * (1 << 20);
}

using sample::gLogError;
using sample::gLogInfo;

//!
//! \class SampleSegmentation
//!
//! \brief Implements semantic segmentation using FCN-ResNet101 ONNX model.
//!
class SampleSegmentation
{

  public:
    SampleSegmentation(const std::string &engineFilename);
    bool infer(const std::string &input_filename, int32_t width, int32_t height, const std::string &output_filename);

  private:
    std::string mEngineFilename; //!< Filename of the serialized engine.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

    // Note that unique_ptr is used to avoid these to be shared among objects.
    std::unique_ptr<nvinfer1::IRuntime> mRuntime;   //!< The TensorRT runtime used to run the network
    std::unique_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
};

// Instantiate inference context and engine from serialized engine file


SampleSegmentation::SampleSegmentation(const std::string &engineFilename)
    : mEngineFilename(engineFilename), mEngine(nullptr)
{
    // De-serialize engine from file
    std::ifstream engineFile(engineFilename, std::ios::binary); // Open engine file as binary
    if (engineFile.fail())
    {
        gLogError << "ERROR: failed to open engine file: " << engineFilename << std::endl;
        return;
    }

    // Get size of file using seekg and tellg (go to end, get current position, go to beginning)
    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    // Read fsize bytes from engine file into engineData
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    // Define runtime context (logger using "this" Logger)
    mRuntime.reset(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));

    // Deserialize engine from engineData in mRuntime context memory
    mEngine.reset(mRuntime->deserializeCudaEngine(engineData.data(), fsize));
    assert(mEngine.get() != nullptr); // Assert engine is not nullptr
}

//!
//! \brief Runs the TensorRT inference.
//!
//! \details Allocate input and output memory, and executes the engine.
//!
bool SampleSegmentation::infer(const std::string &input_filename, int32_t width, int32_t height, const std::string &output_filename)
{   
    // Create CUDA execution context
    auto context = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext()); 
    if (!context)
    {
        return false;
    }


    // Get type of input tensor
    //char const *input_name = "input"; // TIP both input and output name queries in program must be identical to the name given at engine generation for nvinfer to fetch them correctly! Else, you could try to fetch the names using getIOTensorName method

    int nbIOTensors = mEngine->getNbIOTensors();
    std::cout << "Number of I/O tensors: " << nbIOTensors << std::endl;
    // Iterate over each tensor and print its name.
    for (int i = 0; i < nbIOTensors; i++)
    {
        // getIOTensorName returns the name of the tensor at index i.
        const char *tensorName = mEngine->getIOTensorName(i);
        std::cout << "Tensor " << i << ": " << tensorName << std::endl;
    }

    if (nbIOTensors < 2)
    {
        gLogError << "ERROR: Expected at least 2 I/O tensors, but found " << nbIOTensors << std::endl;
        return false;
    }

    char const *input_name = mEngine->getIOTensorName(0);
    std::cout << "Fetching datatype for " << input_name << std::endl;
    std::cout << "Found datatype: " << static_cast<int>(mEngine->getTensorDataType(input_name)) << std::endl;
    
    assert(mEngine->getTensorDataType(input_name) == nvinfer1::DataType::kFLOAT);

    // Specify dimensions of input tensor since dynamic shapes were used to export the model
    auto input_dims = nvinfer1::Dims4{1, /* channels */ 3, height, width};
    context->setInputShape(input_name, input_dims);
    auto input_size = util::getMemorySize(input_dims, sizeof(float)); // Get memory size of input tensor with specified dimensions

    // Do the same for the output tensor
    char const *output_name = mEngine->getIOTensorName(1);
    std::cout << "Fetching datatype for " << output_name << std::endl;
    std::cout << "Found datatype: " << static_cast<int>(mEngine->getTensorDataType(output_name)) << std::endl;
    
    assert(mEngine->getTensorDataType(output_name) == nvinfer1::DataType::kFLOAT);
    auto output_dims = context->getTensorShape(output_name);
    auto output_size = util::getMemorySize(output_dims, sizeof(float));

    // Allocate CUDA memory for input and output bindings
    void *input_mem{nullptr};
    if (cudaMalloc(&input_mem, input_size) != cudaSuccess)
    {
        gLogError << "ERROR: input cuda memory allocation failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }
    void *output_mem{nullptr};
    if (cudaMalloc(&output_mem, output_size) != cudaSuccess)
    {
        gLogError << "ERROR: output cuda memory allocation failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }

    // Read image data from file and mean-normalize it (these are model-specific. In this example the model comes from torchvision, which requires such standardization)
    const std::vector<float> mean{0.485f, 0.456f, 0.406f};
    const std::vector<float> stddev{0.229f, 0.224f, 0.225f};

    std::cout << "Reading input image..." << std::endl;
    auto input_image{util::RGBImageReader(input_filename, input_dims, mean, stddev)}; // Utility class in TensorRT/quickstart/common to read RGB images
    input_image.read();

    auto input_buffer = input_image.process();
    cudaStream_t stream; // Create CUDA stream. Recommended not to use default to avoid performance issues related with default stream CUDA synchronize enforcement.
    if (cudaStreamCreate(&stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA stream creation failed." << std::endl;
        return false;
    }

    // Copy image data to input binding memory (asynch transfer, not blocking)
    if (cudaMemcpyAsync(input_mem, input_buffer.get(), input_size, cudaMemcpyHostToDevice, stream) != cudaSuccess)
    {
        gLogError << "ERROR: CUDA memory copy of input failed, size = " << input_size << " bytes" << std::endl;
        return false;
    }   

    // Bind names of input and output tensors to memory addresses (cuda buffers)
    context->setTensorAddress(input_name, input_mem);
    context->setTensorAddress(output_name, output_mem);

    // Run TensorRT inference
    std::cout << "Running inference..." << std::endl;
    bool status = context->enqueueV3(stream); // NOTE: non blocking call to queue inference task on stream. Returns false if task could not be queued. Alternative call is executeV2
    if (!status)
    {
        gLogError << "ERROR: TensorRT inference failed" << std::endl;
        return false;
    }

    // Copy predictions from output binding memory
    auto output_buffer = std::unique_ptr<float>{new float[output_size]}; // Ptr to buffer

    if (cudaMemcpyAsync(output_buffer.get(), output_mem, output_size, cudaMemcpyDeviceToHost, stream) != cudaSuccess) // Asynch copy from device to host
    {
        gLogError << "ERROR: CUDA memory copy of output failed, size = " << output_size << " bytes" << std::endl;
        return false;
    }
    // Synchronize stream to wait for all operations to finish
    cudaStreamSynchronize(stream);

    // Plot the semantic segmentation predictions of 21 classes in a colormap image and write to file
    const int num_classes{21};
    const std::vector<int> palette{(0x1 << 25) - 1, (0x1 << 15) - 1, (0x1 << 21) - 1};
    auto output_image{util::ArgmaxImageWriter(output_filename, output_dims, palette, num_classes)}; // Utility class in TensorRT/quickstart/common to write images

    float *output_ptr = output_buffer.get();
    std::vector<int32_t> output_buffer_casted(output_size);
    std::cout << "Writing segmented image to file..." << std::endl;

    for (size_t i = 0; i < output_size; ++i)
    {
        output_buffer_casted[i] = static_cast<int32_t>(output_ptr[i]);
    }
    output_image.process(output_buffer_casted.data());
    output_image.write();

    // Free CUDA resources
    cudaFree(input_mem);
    cudaFree(output_mem);
    return true;
}

int main(int argc, char **argv)
{
    int32_t width{1282};
    int32_t height{1026};

    // Get path to engine file
    if (argc < 3)
    {
        gLogError << "Usage: " << argv[0] << " <path to FCN-ResNet101 engine file> <input_image_file> <output_image_file>" << std::endl;
        return -1;
    }

    // Instantiate SampleSegmentation object with engine file
    SampleSegmentation sample(argv[1]);

    gLogInfo << "Running TensorRT inference for FCN-resnet50/101" << std::endl;
    if (!sample.infer(argv[2], width, height, argv[3]))
    {
        gLogError << "ERROR: Inference failed." << std::endl;
        return -1;
    }

    return 0;
}
