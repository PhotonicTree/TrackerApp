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

template <typename T> void RunTrackingForOpenCVLegacyTracker(std::vector<cv::Rect> ROIs, std::vector<cv::Mat> sequence, std::string sequenceName)
{
    std::vector<std::pair<int, cv::Rect>> tracks;
    tracks.reserve(sequence.size());
    JsonTrackerObject jsonTrackerObject;

    auto multiTracker = cv::legacy::MultiTracker::create();
    for (const auto ROI : ROIs)
    {
        multiTracker->add(T::create(), sequence.front(), ROI);
    }
    // Get name of current tracker and push to json file.
    std::string trackerTypeName = typeid(T).name();
    trackerTypeName.erase(0, trackerTypeName.find_last_of(':') + 1);

    for (size_t i = 1; i < sequence.size(); i++)
    {
        std::vector<std::array<int, 4>> frameResult;
        auto image = sequence.at(i); 
        multiTracker->update(image);
        auto& trackerObjects = multiTracker->getObjects();
        for (size_t j = 0; j < trackerObjects.size(); j++)
        {
            auto trackerObject = trackerObjects[j];
            cv::rectangle(image, trackerObject, cv::Scalar(255, 0, 0), 2, 1);
            std::array<int, 4> boundingBox = { static_cast<int>(trackerObject.x), static_cast<int>(trackerObject.y), static_cast<int>(trackerObject.width), static_cast<int>(trackerObject.height) };
            frameResult.push_back(boundingBox);
        }

        // prepare string to json
        std::string frameResultLine = "";
        for (auto bBox : frameResult)
        {
            std::string currentRectangle = "(";
            for (auto number : bBox)
            {
                currentRectangle += std::to_string(number);

                if (number != *(bBox.end() - 1))
                {
                   currentRectangle += ",";
                }
                else
                {
                   currentRectangle += ")";
                }
            }
            frameResultLine += currentRectangle;
            if (bBox != *(frameResult.end() - 1))
            {
               frameResultLine += ",";
            }
        }
        size_t n = 4;
        auto frameNumber = std::to_string(i);
        int precision = n - std::min(n, frameNumber.size());
        frameNumber.insert(0, precision, '0');
        jsonTrackersObject[trackerTypeName][frameNumber] = frameResultLine;
        std::cout << trackerTypeName << " - " << frameNumber << std::endl;
        cv::imshow("Tracker result. Press 'x' to quit", show);
        if (cv::waitKey(100) == 27) break;
    }
    auto tok = std::chrono::high_resolution_clock::now();
    auto durationInSeconds = std::chrono::duration_cast<std::chrono::microseconds>(tok - tik).count() / 1000000.0;
    auto fps = sequence.size() / durationInSeconds;
    jsonTrackersObject[trackerTypeName]["FPS"] = std::to_string(fps);
}

void MultiTrackerMIL::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMIL>(ROIs, sequence, jsonObject);
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

void MultiTrackerMOSSE::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMOSSE>(ROIs, sequence, jsonObject);
}

void MultiTrackerCSRT::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerCSRT::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerCSRT>(ROIs, sequence, jsonObject);
}

void MultiTrackerGOTURN::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerGOTURN::RunTracking(JsonTrackerObject &jsonObject)
{
//    RunTrackingForOpenCVLegacyTracker
}

void MultiTrackerDaSiamRPN::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerDaSiamRPN::RunTracking(JsonTrackerObject &jsonObject)
{
//   RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMIL>(ROIs, sequence, jsonObject);
}

void MultiTrackerTLD::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerTLD::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerTLD>(ROIs, sequence, jsonObject);
}

void MultiTrackerBoosting::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerBoosting::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerBoosting>(ROIs, sequence, jsonObject);
}

void MultiTrackerKCF::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerKCF::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerKCF>(ROIs, sequence, jsonObject);
}

void MultiTrackerMedianFlow::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMedianFlow::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMedianFlow>(ROIs, sequence, jsonObject);
}

void MultiHoughCirclesTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiHoughCirclesTracker::RunTracking(JsonTrackerObject &jsonObject)
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
    circleBlobDetectorParams.thresholdStep = 1;
    circleBlobDetectorParams.minThreshold = 20;
    circleBlobDetectorParams.maxThreshold = 150;
    circleBlobDetectorParams.minRepeatability = 4;
    circleBlobDetectorParams.minDistBetweenBlobs = 10;

    // Filter by color
    circleBlobDetectorParams.filterByColor = true;
    circleBlobDetectorParams.blobColor = 0;

    // Filter by area
    circleBlobDetectorParams.filterByArea = true;
    circleBlobDetectorParams.minArea = 700;
    circleBlobDetectorParams.maxArea = 1500;

    // Filter by circularity
    circleBlobDetectorParams.filterByCircularity = true;
    circleBlobDetectorParams.minCircularity = 0.80F;
    circleBlobDetectorParams.maxCircularity = 1.0F;

    // Filter by convexity
    circleBlobDetectorParams.filterByConvexity = true;
    circleBlobDetectorParams.minConvexity = 0.8F;

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
