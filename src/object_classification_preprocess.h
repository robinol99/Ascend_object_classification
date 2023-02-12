#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>

/**
 * @brief Converts onnx weights, converting them to engine, and saving as saved_name
 * @param model_name Path to onnx input file
 * @param saved_name Name of output engine file
 * @return true if loading successful
 */
bool loadONNX(const std::string& onnx_path, const std::string& saved_name);
