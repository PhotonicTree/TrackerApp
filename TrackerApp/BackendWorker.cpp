#include "BackendWorker.h"

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
	const auto sequencePath = path.toStdString();
	videoCapture.open(sequencePath);

    if (!videoCapture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    cv::Mat frame;
    while (true)
    {
        videoCapture >> frame;
        if (frame.empty())
        {
            break;
        }
        sequence.push_back(frame);
    }
    ImportingFinished();
}

Q_INVOKABLE void BackendWorker::DisplaySequence()
{
    ImageViewer* imageViewer(new ImageViewer);

    imageViewer->TriggerStreamingSequence(videoCapture);
}

Q_INVOKABLE cv::VideoCapture BackendWorker::GetCapture()
{
    return videoCapture;

cv::SimpleBlobDetector::Params InitializeSimpleBlobDetectorParmeters()
{
    cv::SimpleBlobDetector::Params circleBlobDetectorParams;

    // The following parameters are described here:
    // https://docs.opencv.org/4.5.1/d0/d7a/classcv_1_1SimpleBlobDetector.html

    // Filter by intensity (thresholding)
    circleBlobDetectorParams.thresholdStep = 30;
    circleBlobDetectorParams.minThreshold = 50;
    circleBlobDetectorParams.maxThreshold = 231;
    circleBlobDetectorParams.minRepeatability = 2;
    circleBlobDetectorParams.minDistBetweenBlobs = 20;

    // Filter by color
    circleBlobDetectorParams.filterByColor = true;
    circleBlobDetectorParams.blobColor = 0;

    // Filter by area
    circleBlobDetectorParams.filterByArea = true;
    circleBlobDetectorParams.minArea = 80;
    circleBlobDetectorParams.maxArea = 1200;

    // Filter by circularity
    circleBlobDetectorParams.filterByCircularity = true;
    circleBlobDetectorParams.minCircularity = 0.40F;
    circleBlobDetectorParams.maxCircularity = 0.94F;

    // Filter by convexity
    circleBlobDetectorParams.filterByConvexity = true;
    circleBlobDetectorParams.minConvexity = 0.93F;
    circleBlobDetectorParams.maxConvexity = 1.01F;

    // Filter by inertia
    // https://stackoverflow.com/questions/14770756/opencv-simpleblobdetector-filterbyinertia-meaning
    circleBlobDetectorParams.filterByInertia = true;
    circleBlobDetectorParams.minInertiaRatio = 0.25F;
    circleBlobDetectorParams.maxInertiaRatio = 1.01F;

    return circleBlobDetectorParams;
}

}