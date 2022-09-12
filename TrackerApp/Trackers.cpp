#include "Trackers.h"

//template <typename T> void RunOpenCVTrackerByType()
//{
//    std::vector<cv::Ptr<T>> trackingInstances;
//    for (size_t i = 0; i < ROIs.size(); i++)
//    {
//        trackingInstances.push_back(T::create());
//        objects.push_back(ROIs[i]);
//    }
//    this->add(trackingInstances, firstFrame, objects);
//
//    for (const auto& image : sequence)
//    {
//        this->update(image);
//    }
//}

void MultiTrackerMIL::RunTracking()
{
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
    for (size_t i = 0; i < ROIs.size(); i++)
    {
        trackingInstances.push_back(cv::legacy::TrackerMIL::create());
        objects.push_back(ROIs[i]);
    }
    this->add(trackingInstances, firstFrame, objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerMIL::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMOSSE::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
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
    this->add(trackingInstances, firstFrame, objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerCSRT::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
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
    this->add(trackingInstances, firstFrame, objects);

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

void MultiTrackerGOTURN::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

//void MultiTrackerGOTURN::RunTracking()
//{
//    // RunOpenCVTrackerByType<cv::TrackerGOTURN>();
//
//    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
//    for (size_t i = 0; i < ROIs.size(); i++)
//    {
//        trackingInstances.push_back(cv::TrackerGOTURN::create());
//        objects.push_back(ROIs[i]);
//    }
//    this->add(trackingInstances, firstFrame, objects);
//
//    for (const auto& image : sequence)
//    {
//        this->update(image);
//    }
//}

void MultiTrackerDaSiamRPN::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

//void MultiTrackerDaSiamRPN::RunTracking()
//{
//    //RunOpenCVTrackerByType<cv::TrackerDaSiamRPN>();
//    
//    std::vector<cv::Ptr<cv::legacy::Tracker>> trackingInstances;
//    for (size_t i = 0; i < ROIs.size(); i++)
//    {
//        trackingInstances.push_back(cv::TrackerDaSiamRPN::create());
//        objects.push_back(ROIs[i]);
//    }
//    this->add(trackingInstances, firstFrame, objects);
//
//    for (const auto& image : sequence)
//    {
//        this->update(image);
//    }
//}

void MultiTrackerTLD::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
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
    this->add(trackingInstances, firstFrame, objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerBoosting::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
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
    this->add(trackingInstances, firstFrame, objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerKCF::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
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
    this->add(trackingInstances, firstFrame, objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}

void MultiTrackerMedianFlow::InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->firstFrame = image;
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
    this->add(trackingInstances, firstFrame, objects);

    for (const auto& image : sequence)
    {
        this->update(image);
    }
}
