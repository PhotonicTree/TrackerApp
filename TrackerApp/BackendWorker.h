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
		GOTURN,
		DaSiamRPN,
		MIL,
		TLD,
		Boosting,
		KCF,
		MedianFlow,
		HoughCircles,
		BlobDetector
	};
}

/*! \class BackendWorker
	Main class in TrackerApp for performing all back-end operations.
*/
class BackendWorker : public QObject
{
	Q_OBJECT

signals:
	//! Signal used to emit, when importing is finished.
	void ImportingFinished();

public slots:
	Q_INVOKABLE void ImportSequence(const QString& path);
	Q_INVOKABLE void DisplaySequence();
	Q_INVOKABLE void GetFirstFrameROI();
	Q_INVOKABLE void GetSelectedTrackers(std::vector<bool> selectedTrackersQML);
	Q_INVOKABLE void RunAllTrackers();

private:
	void HoughCirclesTracker();
	void DetectMarkersOnImage();
	void RunTrackerCSRT();
	void RunTrackerDaSiamRPN();
	void RunTrackerGOTURN();
	void RunTrackerMOSSE();

	std::vector<cv::Mat> sequence;
	cv::VideoCapture videoCapture;
	cv::Rect ROI;
	std::map<BackendWorkerHelpers::TrackerType, bool> selectedTrackers;
};


