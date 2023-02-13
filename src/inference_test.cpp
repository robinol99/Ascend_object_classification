#include "object_classification_preprocess.h"
#include "utilities/utilities_nvidia.h"
#include <numeric>
#include <opencv2/opencv.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

void* bufferPtr[5];
nvinfer1::ICudaEngine* engine = nullptr;

void preprocess_img();
void postproces_img();

int main(){
    std::string engine_path = "/home/robinol/git_projects/onnx_tensorrt/classification_engine_output_w_softmax.engine";
    
    bool eng = AscendUtils::loadEngine(&engine, engine_path);
    std::cout << eng << std::endl;
    
    nvinfer1::IExecutionContext* engineContext = engine->createExecutionContext();


    for (int i = 0; i < engine->getNbBindings(); i++){
        nvinfer1::Dims tensor = engine->getBindingDimensions(i);

        size_t size = std::accumulate(tensor.d+1, tensor.d+tensor.nbDims, 1, std::multiplies<size_t>());

        cudaMalloc(&bufferPtr[i], size * sizeof(float));

        bool input = engine->bindingIsInput(i);
        printf("Found tensor: %i, input: %i, name: %s, size %i\n", i, input, engine->getBindingName(i), size);


    }

    preprocess_img();
    engineContext->executeV2(bufferPtr);
    cudaDeviceSynchronize();
    postproces_img();

    for (int i = 0; i < 5; i++){
        cudaFree(bufferPtr[i]);
    }
    
    delete engineContext;
    delete engine;
    return 0;
}


void preprocess_img(){
    std::string img_loc = "/media/nas_datasets/suas23_object_classification/images_v3/0000000100.png";

    cv::Mat img = cv::imread(img_loc, cv::IMREAD_COLOR);
    //cv::imshow("Display window25", img);
    //int k10 = cv::waitKey(0); // Wait for a keystroke in the window

    if(img.empty())
    {
        std::cout << "Could not read the image: " << img_loc << std::endl;
        return;
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
        chw.emplace_back(cv::cuda::GpuMat(input_size, CV_32FC1, bufferPtr[0] + i * input_width * input_height));
    }
    cv::cuda::split(flt_image, chw);


    cudaDeviceSynchronize();

    //cv::Mat img3;
    //flt_image.download(img3);

    //std::cout << "img3 rows: " << img3.rows << std::endl;
    //std::cout << "img3 cols: " << img3.cols << std::endl;

    //cv::imshow("Display window1", roi);
    //int k = cv::waitKey(0); // Wait for a keystroke in the window
    //cv::imshow("Display window2", img3);
    //int k2 = cv::waitKey(0); // Wait for a keystroke in the window

}

void postproces_img(){
    for (int i = 1; i < 5; i++){
        nvinfer1::Dims dims = engine->getBindingDimensions(i);

        size_t dimsSize = std::accumulate(dims.d+1, dims.d+dims.nbDims, 1, std::multiplies<size_t>());
        std::vector<float> cpu_output (dimsSize);

        cudaMemcpy(cpu_output.data(), bufferPtr[i], cpu_output.size()*sizeof(float), cudaMemcpyDeviceToHost);

        int n_eitlanan = dims.d[1];
        std::cout << "cpu output size: " << cpu_output.size() << std::endl;

        std::cout << "Output for " << i << ": ";
        for (int j = 0; j < n_eitlanan; j++){
            float nesteVerdi = cpu_output[j];
            std::cout << nesteVerdi << " ";
        }
        std::cout << std::endl;
    }
}

