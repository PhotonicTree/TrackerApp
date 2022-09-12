#include "Trackers.h"

BaseTracker::BaseTracker()
{
}

BaseTracker::~BaseTracker()
{
}

void MultiTrackerMIL::RunTracking()
{
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerMIL::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerMIL::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{ 
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMOSSE::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMOSSE::RunTracking()
{
    //RunOpenCVTrackerByType<cv::legacy::tracking::TrackerMOSSE>();

    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerMOSSE::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerCSRT::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerCSRT::RunTracking()
{
   // RunOpenCVTrackerByType<cv::tracking::TrackerCSRT>();

    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerCSRT::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);

        for (auto i = 0; i < this->getObjects().size(); i++)
        {
            cv::rectangle(image, this->getObjects()[i], cv::Scalar(255, 0, 0), 2, 1);
        }
        cv::imshow("tracker", image);
    }
}

void MultiTrackerGOTURN::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerGOTURN::RunTracking()
{
//    // RunOpenCVTrackerByType<cv::TrackerGOTURN>();
//
//    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
//    for (size_t i = 0; i < ROIs.size(); i++)
//    {
//        trackingInstances.push_back(cv::TrackerGOTURN::create());
//        objects.push_back(ROIs[i]);
//    }
//    this->add(trackingInstances, sequence.front(), objects);
//
//    for (const auto& image : sequence)
//    {
//        this->update(image);
//    }
}

void MultiTrackerDaSiamRPN::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerDaSiamRPN::RunTracking()
{
//    //RunOpenCVTrackerByType<cv::TrackerDaSiamRPN>();
//    
//    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
//    for (size_t i = 0; i < ROIs.size(); i++)
//    {
//        trackingInstances.push_back(cv::TrackerDaSiamRPN::create());
//        objects.push_back(ROIs[i]);
//    }
//    this->add(trackingInstances, sequence.front(), objects);
//
//    for (const auto& image : sequence)
//    {
//        this->update(image);
//    }
}

void MultiTrackerTLD::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerTLD::RunTracking()
{
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerTLD::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerBoosting::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerBoosting::RunTracking()
{
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerBoosting::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerKCF::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerKCF::RunTracking()
{
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerKCF::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerMedianFlow::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMedianFlow::RunTracking()
{
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerMedianFlow::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, sequence.front(), objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiBlobDetectorTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->circleRadius = radius;

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
}

void MultiBlobDetectorTracker::RunTracking()
{
    auto blobDetector = cv::SimpleBlobDetector::create(circleBlobDetectorParams);
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    for (auto i = 0; i < sequence.size(); ++i)
    {
        blobDetector->detect(sequence[i], keypoints[i]);

        for (size_t j = 0; j < keypoints.size(); j++)
        {
            cv::Point center(cvRound(keypoints[i][j].pt.x), cvRound(keypoints[i][j].pt.y));
            // draw the circle center
            cv::circle(sequence[i], center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // draw the circle outline
            circle(sequence[i], center, circleRadius, cv::Scalar(0, 0, 255), 3, 8, 0);
        }
    }
}
