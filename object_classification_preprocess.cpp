#include "object_classification_preprocess.h"
#include "utilities/utilities_nvidia.h"

static AscendUtils::NvidiaLogger logger;

bool loadONNX(const std::string& onnx_path, const std::string& saved_name){
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);

    uint32_t flag = 1U <<static_cast<uint32_t>
    (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);

    bool success = parser->parseFromFile(onnx_path.c_str(),
                                         static_cast<int32_t>(nvinfer1::ILogger::Severity::kINFO));

    if (!success){
        std::cout << "Failed to parse onnx from file " << onnx_path << "\n";
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

    // 4 GB // TODO - make this a parameter
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, (4 * 1U) << 30);

//    IOptimizationProfile* profile = builder->createOptimizationProfile();
//    //Model 1x 3 channels x size
//    profile->setDimensions("input", OptProfileSelector::kMAX, Dims4(1, 3, 640, 640));
//    profile->setDimensions("input", OptProfileSelector::kMIN, Dims4(1, 3, 640, 640));
//    profile->setDimensions("input", OptProfileSelector::kOPT, Dims4(1, 3, 640, 640));
//    config->addOptimizationProfile(profile);

    nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

    std::ofstream write_out(saved_name, std::ios::binary);
    write_out.write((const char*) serializedModel->data(), serializedModel->size());
    write_out.close();

    delete parser;
    delete network;
    delete config;
    delete builder;

    return true;
}

int main(int argc, char** argv){
    if (argc < 3){
        std::printf("Usage: ./object_classification_preprocess {input.onnx} {output.engine}\n");
        return 1;
    }

    bool success = loadONNX(std::string(argv[1]), std::string(argv[2]));

    if (!success){
        std::printf("ONNX parsing failed...\n");
        return 1;
    }
    return 0;
}
