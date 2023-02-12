#pragma once
#include <NvInfer.h>
#include <boost/filesystem.hpp>

namespace AscendUtils{
    /**
     * @brief Logger class for Nvidia TensorRT
     */
    class NvidiaLogger : public nvinfer1::ILogger{
        void log(Severity severity, const char* msg) noexcept override;
    };

    /**
     * @brief Loads .engine optimized weights
     * @param engine ICudaEngine object to load the file into
     * @param model_name Path to engine file
     * @return true if loading successful
     */
    bool loadEngine(nvinfer1::ICudaEngine** engine, const boost::filesystem::path& model_path);
}