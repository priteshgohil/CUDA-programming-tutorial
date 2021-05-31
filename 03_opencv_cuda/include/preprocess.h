/*
 * preprocess.h
 *
 *  Created on: 28.05.2021
 *      Author: pgohil
 */

#ifndef INCLUDES_PREPROCESS_H_
#define INCLUDES_PREPROCESS_H_

#include <iostream>
#include <opencv2/opencv.hpp>

class Preprocess {
    private:
        double dBinaryThreshold = 128.0;
        double dThresholdMax = 255.0;
        int nthresholdType = cv::THRESH_BINARY;
 
    public:
	    void set_binary_threshold_cpu(cv::Mat &src, cv::Mat &dst);
        //void set_binary_threshold_gpu(cv::gpu::Mat &cuda_src, cv::gpu::Mat &cuda_dst);

};



#endif /* INCLUDES_PREPROCESS_H_ */
