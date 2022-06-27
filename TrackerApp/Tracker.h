#pragma once
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>

using MarkerPosition = std::pair<size_t, std::pair<double, double>>;
using MarkerTrace = std::vector<MarkerPosition>;

class Tracker
{
private:
	//cv::Ptr<cv::SimpleBlobDetector> trackerBlobDetector; //!< Pointer to SimpleBlobDetector.
	//std::shared_ptr<OpenCVMosseTracker> mosseTracker;	 //!< Pointer to OpenCV MOSSE tracker.
	MarkerTrace markerTrace;							 //!< Marker trace tracked by tracker.
	int initialBoundingBoxWidth = 140;					 //!< Width of initial bounding box for tracker
	int initialBoundingBoxHeight = 140;					 //!< Height of initial bounding box for tracker

public:
	/*! Tracker constructor.
		\param[in] initPosition Initial position of marker which has to be tracked. Based on this position initial ROI is set.
	*/
	Tracker(MarkerPosition initPosition);
	/*! Tracker constructor.
		\param[in] initPosition Initial position of marker which has to be tracked. Based on this position initial ROI is set.
		\param[in] blobDetectorParamsFilePath Path to file with parameters for SimpleBlobDetector.
	*/
	Tracker(MarkerPosition initPosition, const std::filesystem::path& blobDetectorParamsFilePath);

	/*! Method used to get marker trace.
		\returns Marker trace tracked by tracker.
	*/
	MarkerTrace GetMarkerTrace();

	/*! Method used to track marker by the set number of frames.
		\param[in] inputImageSequence All sequence frames.
		\param[in] beginFrameIndex Index of frame from which tracker has to track marker.
		\param[in] numberOfFramesToProcess Number of frames for which the tracker has to work.
	*/
	void Track(std::vector<cv::Mat>& inputImageSequence, size_t beginFrameIndex, size_t numberOfFramesToProcess);
};