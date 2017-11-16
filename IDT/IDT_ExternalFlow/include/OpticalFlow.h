#ifndef OPTICALFLOW_H_
#define OPTICALFLOW_H_

#include "DenseTrackStab.h"

#include <time.h>

using namespace cv;

namespace my
{
    // convert flow image to flow matrix by de-normalizing
    void convertImageToFlow(const Mat& image, Mat& flow, double lowerBound, double higherBound)
    {
        for(int i = 0; i < image.rows; ++i)
            for(int j = 0; j < image.cols; ++j)
                flow.at<float>(i,j) = image.at<float>(i,j) * (higherBound - lowerBound) / 255.0 + lowerBound;
    }
}

#endif /*OPTICALFLOW_H_*/
