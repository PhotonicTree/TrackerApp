#include "Trackers.h"
#include "JsonTrackerObject.h"
#include <string>
#include <fstream>

BaseTracker::BaseTracker()
{
}

BaseTracker::~BaseTracker()
{
}

template <typename T> void RunTrackingForOpenCVLegacyTracker(std::vector<cv::Rect> ROIs, std::vector<cv::Mat> sequence)
{
    auto multiTracker = cv::legacy::MultiTracker::create();
    for (const auto ROI : ROIs)
    {
        multiTracker->add(T::create(), sequence.front(), ROI);
    }

    for (size_t i = 1; i < sequence.size(); i++)
    {
        auto image = sequence.at(i);
        multiTracker->update(image);
        auto& trackerObjects = multiTracker->getObjects();
        for (size_t i = 0; i < trackerObjects.size(); i++)
        {
            cv::rectangle(image, trackerObjects[i], cv::Scalar(255, 0, 0), 2, 1);
        }
        cv::imshow("Tracker result. Press 'x' to quit", image);
        if (cv::waitKey(100) == 27) break;
    }
}

void MultiTrackerMIL::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMIL>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerMIL::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{ 
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerMOSSE::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerMOSSE::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMOSSE>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerCSRT::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerCSRT::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerCSRT>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerGOTURN::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerGOTURN::RunTracking()
{
//    RunTrackingForOpenCVLegacyTracker
}

void MultiTrackerDaSiamRPN::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerDaSiamRPN::RunTracking()
{
//   RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMIL>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerTLD::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerTLD::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerTLD>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerBoosting::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerBoosting::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerBoosting>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerKCF::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerKCF::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerKCF>(ROIs, sequence, sequenceFileName);
}

void MultiTrackerMedianFlow::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiTrackerMedianFlow::RunTracking()
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMedianFlow>(ROIs, sequence, sequenceFileName);
}

void MultiHoughCirclesTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->sequenceFileName = sequenceName;
}

void MultiHoughCirclesTracker::RunTracking()
{
    for (const auto& frame : sequence)
    {
        for (const auto& ROI : ROIs)
        {
            cv::Mat3b frame3b = frame.clone();
            std::vector<cv::Vec3f> circles;
            cv::Mat grayImage;
            auto ROIImage = frame3b(ROI);
            cv::cvtColor(ROIImage, grayImage, cv::COLOR_BGR2GRAY);
            cv::medianBlur(grayImage, grayImage, 5);
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
            cv::imshow("HoughCircles Result", frame3b);
            if (cv::waitKey(100) == 27) break;
        }
    }
}

void MultiBlobDetectorTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, std::string sequenceName, int radius)
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
    std::vector<std::vector<cv::KeyPoint>> keypoints(100);
    for (auto i = 0; i < sequence.size(); ++i)
    {
        blobDetector->detect(sequence.at(i), keypoints.at(i));

        for (size_t j = 0; j < keypoints.at(i).size(); ++j)
        {
            cv::Point center(cvRound(keypoints.at(i).at(j).pt.x), cvRound(keypoints.at(i).at(j).pt.y));
            // draw the circle center
            cv::circle(sequence[i], center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
            // draw the circle outline
            circle(sequence[i], center, circleRadius, cv::Scalar(0, 0, 255), 3, 8, 0);
        }
    }
}
