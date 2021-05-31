//============================================================================
// Name        : opencv_cuda.cpp
// Author      : Pritesh
// Version     :
// Copyright   : Your copyright notice
// Description : OpenCV project to demonstrate live streaming using CPU & CUDA
//============================================================================

#include <iostream>
#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
//#include <opencv2/cudafilters.hpp> 
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>

// Add user defined input files
#include <global.h>
#include <common.h>
#include <preprocess.h>

int main(int argc, char** argv) {
    bool bWebcam = false;
    unsigned int unCam;
    Common fn;
    Preprocess preprocess;
    
    // for host and device
    cv::Mat hostImgSrc, hostImgDst;
    //cv::gpu::GpuMat deviceImgSrc, deviceImgDst;

    // Parse arguments
    if(argc == 3 && std::string(argv[1]) == "-d") {
	// check if argument is numeric i.e. webcame device number
	bool bIsNumericArg = fn.is_number(argv[2]);
	if (bIsNumericArg == true) {
	    bWebcam = true;
	    unCam = std::stoi(argv[2]);
	    std::cout << "Selecting webcam - " << argv[2] << std::endl;
	}
    }

    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    if (bWebcam == true) {
        cv::VideoCapture cap(unCam);
        if (!cap.isOpened()) {
            std::cerr << "ERROR: Could not open camera: " << unCam << std::endl;
            return 1;
        }

        

        // Set webcam resolution - original resolution will reduce the total FPS
        cap.set(cv::CAP_PROP_FRAME_WIDTH, DEFAULT_FRAME_WIDTH);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, DEFAULT_FRAME_HEIGHT);

        // Query resolution
        cap >> hostImgSrc;
        cv::Size sSize = hostImgSrc.size();
        int w = sSize.width;
        //int h = sSize.height;

        cv::namedWindow("Webcam", cv::WINDOW_AUTOSIZE);
        std::cout << "Starting webcame, press Esc to exit " << std::endl;
        // display the frame until you press Esc key
        while (1) {
            auto t_start = std::chrono::system_clock::now();
            // capture the next frame from the webcam
            cap >> hostImgSrc;
            // apply threshold using CPU
            preprocess.set_binary_threshold_cpu(hostImgSrc, hostImgDst);
            // apply threshold using GPU to speedup
            // 1. copy frame host to device
            //deviceImgSrc.upload(hostImgSrc);
            // 2. perform operation on device 
            //preprocess.set_binary_threshold_cpu(deviceImgSrc, deviceImgDst);
            // 3. copy frame device to host
            //deviceImgDst.download(hostImgDst);
   
            // show the image on the window
            cv::imshow("Webcam", hostImgDst);
            // wait (10ms) for Esc key to be pressed
            if (cv::waitKey(10) == 27)
                break;
	    auto t_end = std::chrono::system_clock::now();
	    // Measure frame read + display time 
            std::cout << "Camera read and display duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count() << "ms" << std::endl;
        }
    }
    return 0;
}
