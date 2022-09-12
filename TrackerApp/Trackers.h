#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cvstd.hpp>


inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
	cv::Ptr<cv::Tracker> tracker;
	//cv::Ptr<cv::legacy::Tracker> tracker;

	if (name == "KCF")
		tracker = cv::TrackerKCF::create();
	//else if (name == "TLD")
	//{
	//	auto legacyTracker = cv::TrackerKCF::create();
	//	tracker = cv::Ptr<cv::Tracker>::dynamicCast(legacyTracker);
	//}
	else if (name == "BOOSTING")
		//tracker = cv::legacy::TrackerBoosting::create();
	//else if (name == "MEDIAN_FLOW")
		//tracker = cv::legacy::TrackerMedianFlow::create();
	//else if (name == "MIL")
		tracker = cv::TrackerMIL::create();
	else if (name == "GOTURN")
		tracker = cv::TrackerGOTURN::create();
	//else if (name == "MOSSE")
		//tracker = cv::legacy::TrackerMOSSE::create();
	else
		CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

	return tracker;
}

class BaseTracker : public cv::legacy::MultiTracker
{
public:
	//BaseTracker();
	//BaseTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	//~BaseTracker();

	virtual void RunTracking() = 0;
	std::vector<cv::Rect> ROIs;
	cv::Mat firstFrame;
	std::vector<cv::Mat> sequence;

class MultiBlobDetectorTracker : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
		\param[in] radius int value of circle to be drawn around found marker.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius = 10);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking();

private:
	cv::SimpleBlobDetector::Params circleBlobDetectorParams;
	int circleRadius;
};

class MultiTrackerMIL : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiHoughCirclesTracker : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking();
};

class MultiTrackerCSRT : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiTrackerGOTURN : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiTrackerDaSiamRPN : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiTrackerTLD : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiTrackerBoosting : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiTrackerKCF : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};

class MultiTrackerMedianFlow : public BaseTracker
{
public:
	void InitializeTracker(cv::Mat& image, std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);
	virtual void RunTracking();
};
