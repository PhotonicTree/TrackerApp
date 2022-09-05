#include "BackendWorker.h"

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
	const auto sequencePath = path.toStdString();
	videoCapture.open(sequencePath);

    if (!videoCapture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    while (videoCapture.isOpened())
    {
        cv::Mat frame;
        videoCapture >> frame;
        if (frame.empty())
        {
            break;
        }
        sequence.push_back(frame);
    }
    videoCapture.release();
    ImportingFinished();
}

Q_INVOKABLE void BackendWorker::DisplaySequence()
{
    ImageViewer* imageViewer(new ImageViewer);

    imageViewer->TriggerStreamingSequence(videoCapture);
}

Q_INVOKABLE void BackendWorker::GetFirstFrameROI()
{
    auto& firstFrame = sequence.front();
    ROI = cv::selectROI("tracker", firstFrame);
    std::cout << "Roi selected";
}

Q_INVOKABLE void BackendWorker::GetSelectedTrackers(std::vector<bool> selectedTrackersQML)
{
    for (auto i = 0; i < selectedTrackersQML.size(); ++i)
    {
        selectedTrackers.insert(std::make_pair(static_cast<BackendWorkerHelpers::TrackerType>(i), selectedTrackersQML[i]));
    }
}

Q_INVOKABLE void BackendWorker::RunAllTrackers()
{ 
    for (const auto selectedTracker : selectedTrackers)
    {
        switch (selectedTracker.first) {
        case BackendWorkerHelpers::TrackerType::CSRT: RunTrackerCSRT();
            break;
        /*case BackendWorkerHelpers::TrackerType::MOSSE: RunTrackerMOSSE();
            break;*/
        case BackendWorkerHelpers::TrackerType::DaSiamRPN: RunTrackerDaSiamRPN();
            break;
        case BackendWorkerHelpers::TrackerType::GOTURN: RunTrackerGOTURN();
            break;
        //case BackendWorkerHelpers::TrackerType::CSRT: RunTrackerCSRT();
        //    break;
        //case BackendWorkerHelpers::TrackerType::CSRT: RunTrackerCSRT();
        //    break;
        }
    }
    // RunTrackeDaSiamRPN();
}

void BackendWorker::HoughCirclesTracker()
{
    for (const auto& frame : sequence)
    {
        std::vector<cv::Vec3f> circles;
        cv::Mat grayImage;
        cv::Mat canny;
        auto ROIImage = frame(ROI);
        cv::cvtColor(ROIImage, grayImage, cv::COLOR_BGR2GRAY);
        cv::medianBlur(grayImage, grayImage, 5);
       // cv::Canny(grayImage, canny, 160, 0, 3);
        cv::HoughCircles(grayImage, circles, cv::HOUGH_GRADIENT, 1, grayImage.rows / 16, 140, 10, 1, 10);

        for (size_t i = 0; i < circles.size(); i++)
        {
            cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            int radius = cvRound(circles[i][2]);
            // draw the circle center
            cv::circle(ROIImage, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // draw the circle outline
            circle(ROIImage, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
        }
        cv::namedWindow("circles", 1);
        cv::imshow("circles", ROIImage);
        cv::waitKey(0);
    }
}

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

void BackendWorker::DetectMarkersOnImage()
{
    auto firstFrame = sequence.front();
    auto blobDetector = cv::SimpleBlobDetector::create(InitializeSimpleBlobDetectorParmeters());
    std::vector<cv::KeyPoint> keypoints;
    blobDetector->detect(firstFrame, keypoints);

    for (size_t i = 0; i < keypoints.size(); i++)
    {
        cv::Point center(cvRound(keypoints[i].pt.x), cvRound(keypoints[i].pt.y));
        int radius = 10;
        // draw the circle center
        cv::circle(firstFrame, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
        // draw the circle outline
        circle(firstFrame, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
    }

}

void BackendWorker::RunTrackerCSRT()
{

}