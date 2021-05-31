//============================================================================
// Name        : preprocess.cpp
// Author      : Pritesh
// Version     :
// Copyright   : Your copyright notice
// Description : Image preprocessing functions here
//============================================================================

#include <preprocess.h>

void Preprocess::set_binary_threshold_cpu(cv::Mat &src, cv::Mat &dst) {
    cv::threshold(src, dst, this->dBinaryThreshold, this->dThresholdMax, this->nthresholdType);
}

