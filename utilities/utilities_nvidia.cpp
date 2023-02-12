#include "utilities_nvidia.h"
#include <iostream>

bool AscendUtils::loadEngine(nvinfer1::ICudaEngine** engine, const boost::filesystem::path& model_path){
    // TODO - maybe instead return engine and return nullptr if failed
    static NvidiaLogger logger;

    std::ifstream opened_file_stream(model_path.string(), std::ios::in | std::ios::binary);

    std::printf("Loading network from %s\n", model_path.string().c_str());


    opened_file_stream.seekg(0, std::ios::end);
    const auto model_size = opened_file_stream.tellg();
    opened_file_stream.seekg(0, std::ios::beg);

    std::cout << "Model size is " << model_size << std::endl;
//    std::printf("Model size is %d\n", model_size);

    void* model_mem = malloc(model_size);
    if (!model_mem){
        std::printf("Failed to allocate %i bytes to deserialize model\n", model_size);
        return false;
    }
    opened_file_stream.read((char*)model_mem, model_size);


    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);

    *engine = runtime->deserializeCudaEngine(model_mem, model_size);

    free(model_mem);
    delete runtime;

    if (!*engine){
        std::cout << "Engine creation failed\n";
        return false;
    }

    return true;
}


void AscendUtils::NvidiaLogger::log(nvinfer1::ILogger::Severity severity, const char *msg) noexcept {
    switch (severity){
        case Severity::kINTERNAL_ERROR:
            std::cout << "INTERNAL_ERROR: " << msg << std::endl;
            break;
        case Severity::kERROR:
            std::cout << "ERROR: " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cout << "WARNING: " << msg << std::endl;
            break;
        case Severity::kINFO:
            std::cout << "INFO: " << msg << std::endl;
            break;
        default:
            std::cout << "UNKNOWN: " << msg << std::endl;
            break;
    }
}