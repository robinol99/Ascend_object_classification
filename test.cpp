#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <string>


void preprocessImage(const std::string& image_path, float* gpu_input){ //const nvinfer1::Dims& dims){
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::cerr << "Input image " << image_path << " load failed\n";
        return;
    }
    cv::Mat roi(frame, cv::Rect(64,0,369,369)); //For pictures with same resolution as in datasets.

    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(roi);

    auto input_width = 224; //dims.d[2];
    auto input_height = 224; //dims.d[1];
    auto channels = 3; //dims.d[0];
    auto input_size = cv::Size(input_width, input_height);
    
    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);

    std::vector<cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);
}

/*
void postprocessResults(float *gpu_output, const nvinfer1::Dims &dims, int batch_size){
    // get class names
    auto classes = getClassNames("imagenet_classes.txt");
 
    // copy results from GPU to CPU
    std::vector< float > cpu_output(getSizeByDim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), gpu_output, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    // calculate softmax
    std::transform(cpu_output.begin(), cpu_output.end(), cpu_output.begin(), [](float val) {return std::exp(val);});
    auto sum = std::accumulate(cpu_output.begin(), cpu_output.end(), 0.0);
    // find top classes predicted by the model
    std::vector< int > indices(getSizeByDim(dims) * batch_size);
    // generate sequence 0, 1, 2, 3, ..., 999
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&cpu_output](int i1, int i2) {return cpu_output[i1] > cpu_output[i2];});
    // print results
    int i = 0;
    while (cpu_output[indices[i]] / sum > 0.005)
    {
        if (classes.size() > indices[i])
        {
            std::cout << "class: " << classes[indices[i]] << " | ";
        }
        std::cout << "confidence: " << 100 * cpu_output[indices[i]] / sum << "% | index: " << indices[i] << "n";
        ++i;
    }
}
}
*/

class Logger : public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override {
        // remove this 'if' if you need more logged info
        if ((severity == Severity::kVERBOSE) || (severity == Severity::kINTERNAL_ERROR)) {
            std::cout << msg << "n";
        }
    }
};

Logger gLogger;



size_t getSizeByDim(const nvinfer1::Dims& dims)
{
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i)
    {
        size *= dims.d[i];
    }
    return size;
}


std::vector< std::string > getClassNames(const std::string& model_classes)
{
    std::ifstream classes_file(model_classes);
    std::vector< std::string > classes;
    if (!classes_file.good())
    {
        std::cerr << "ERROR: can't read file with classes names.n";
        return classes;
    }
    std::string class_name;
    while (std::getline(classes_file, class_name))
    {
        classes.push_back(class_name);
    }
    return classes;
}



/*
struct TRTDestroy
{
    template< class T >
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};
 
template< class T >
using TRTUniquePtr = std::unique_ptr< T, TRTDestroy >;




void parseOnnxModel(const std::string& model_path, TRTUniquePtr<nvinfer1::ICudaEngine>& engine,
                    TRTUniquePtr< nvinfer1::IExecutionContext >& context){
    TRTUniquePtr< nvinfer1::IBuilder > builder{nvinfer1::createInferBuilder(gLogger)};
    TRTUniquePtr< nvinfer1::INetworkDefinition > network{builder->createNetwork()};
    TRTUniquePtr< nvonnxparser::IParser > parser{nvonnxparser::createParser(*network, gLogger)};
    // parse ONNX
    if (!parser->parseFromFile(model_path.c_str(), static_cast< int >(nvinfer1::ILogger::Severity::kINFO)))
    {
        std::cerr << "ERROR: could not parse the model.\n";
        return;
    }

    TRTUniquePtr< nvinfer1::IBuilderConfig > config{builder->createBuilderConfig()};
    // allow TensorRT to use up to 1GB of GPU memory for tactic selection.
    config->setMaxWorkspaceSize(1ULL << 30);
    // use FP16 mode if possible
    if (builder->platformHasFastFp16())
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    // we have only one image in batch
    builder->setMaxBatchSize(1);
    engine.reset(builder->buildEngineWithConfig(*network, *config));
        context.reset(engine->createExecutionContext());
}

*/
int main(){
    std::string ONNX_FILE_PATH= "classification_model.onnex";
    std::string img_loc = "/media/nas_datasets/suas23_object_classification/images_v3/0000000100.png";
    std::cout << img_loc << std::endl;
    

    return 0;
}





/*

    cv::Mat img = cv::imread(img_loc, cv::IMREAD_COLOR);

    if(img.empty())
    {
        std::cout << "Could not read the image: " << img_loc << std::endl;
        return 1;
    }

    cv::Mat roi(img, cv::Rect(64,0,369,369));
    std::cout << "rows: " << roi.rows << std::endl;
    std::cout << "cosl: " << roi.cols << std::endl;

    cv::cuda::GpuMat gpu_frame;
    // upload image to GPU
    gpu_frame.upload(roi);

    auto input_width = 224;
    auto input_height = 224;
    auto channels = 3;
    auto input_size = cv::Size(input_width, input_height);

    // resize
    cv::cuda::GpuMat resized;
    cv::cuda::resize(gpu_frame, resized, input_size, 0, 0, cv::INTER_NEAREST);

    std::cout << "resized rows: " << resized.rows << std::endl;
    std::cout << "resized cols: " << resized.cols << std::endl;

    cv::cuda::GpuMat flt_image;
    resized.convertTo(flt_image, CV_32FC3, 1.f / 255.f);
    cv::cuda::subtract(flt_image, cv::Scalar(0.485f, 0.456f, 0.406f), flt_image, cv::noArray(), -1);
    cv::cuda::divide(flt_image, cv::Scalar(0.229f, 0.224f, 0.225f), flt_image, 1, -1);


    std::vector<cv::cuda::GpuMat > chw;
    for (size_t i = 0; i < channels; ++i)
    {
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, gpu_input + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);

    cv::Mat img2;
    flt_image.download(img2);

    std::cout << "img2 rows: " << img2.rows << std::endl;
    std::cout << "img2 cols: " << img2.cols << std::endl;

    cv::imshow("Display window", roi);
    int k = cv::waitKey(0); // Wait for a keystroke in the window
    cv::imshow("Display window", img2);
    int k2 = cv::waitKey(0); // Wait for a keystroke in the window

    */