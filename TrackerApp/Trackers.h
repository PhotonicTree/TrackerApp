#pragma once
#include "JsonTrackerObject.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

class BaseTracker
{
public:
	BaseTracker();
	~BaseTracker();

	/*Pure virtual function dedicated to inheriting classes for performing tracking.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject) = 0;
	
protected:
	std::vector<cv::Rect> ROIs;			//Vector of cv::Rect with selected regions of interests by user.
	std::vector<cv::Mat> sequence;		//Vector with sequence of images to be analyze.
	std::string sequenceFileName;
};

class MultiBlobDetectorTracker : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
		\param[in] radius int value of circle to be drawn around found marker.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius = 20);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	void RunTracking(JsonTrackerObject &jsonObject) override;

private:
	cv::SimpleBlobDetector::Params circleBlobDetectorParams;
	int circleRadius;
};

class MultiTrackerMIL : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
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
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerMOSSE : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerCSRT : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class TrackerGOTURN : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class TrackerDaSiamRPN : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerTLD : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerBoosting : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerKCF : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerMedianFlow : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
