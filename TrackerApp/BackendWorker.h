#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core_detect.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <iostream>

#include "ImageViewer.h"
#include  "Trackers.h"

/*!	\file BackendController.h
	This file contains BackendWorker class, which is responsible for the whole back-end operations.
*/
 
/*Namespace for placing all needed helpers in Backed Worker.*/
namespace BackendWorkerHelpers
{
	/*Enum class representing all types of available trackers.*/
	enum class TrackerType
	{
		MOSSE,
		CSRT,
		MIL,
		TLD,
		Boosting,
		KCF,
		MedianFlow,
		HoughCircles,
		BlobDetector,
		GOTURN,
		DaSiamRPN
	};
}

/*Main class in TrackerApp for performing all back-end operations.*/
class BackendWorker : public QObject
{
	Q_OBJECT

signals:
	//Signal used to emit, when importing is finished.
	void importingFinished();

	//Signal used to emit, when importing is finished.
	void importingFailed();

public slots:

	/*Method used to import sequence of images.
		\param[in] path QString with path to the video.
	*/
	Q_INVOKABLE void ImportSequence(const QString& path);

	/*Method used to display loaded sequence.*/
	Q_INVOKABLE void DisplaySequence();

	/*Method used to get informations from GUI switches which tracker types are selected.
		\param[in] selectedTrackersQML std::vector<bool> with states of each tracker types.
	*/
	Q_INVOKABLE void GetSelectedTrackers(std::vector<bool> selectedTrackersQML);

	/*Method used to run all selected trackers.*/
	Q_INVOKABLE void RunAllTrackers();

private:
	/*Method used to allow user to select markers' ROIs.*/
	void GetFirstFrameROIs();

	std::vector<cv::Mat> sequence;		// Vector with sequence of images to be analyzed.
	cv::VideoCapture videoCapture;		// VideoCapture loaded by user.
	std::vector<cv::Rect> ROIs;			// Vector of cv::Rect with selected regions of interests by user.
	std::map<BackendWorkerHelpers::TrackerType, bool> selectedTrackers;		// Map with tracker types as a key and them boolean state as value.
	std::string sequenceName;
};


