#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core_detect.hpp>

#include <iostream>

#include "ImageViewer.h"


class BackendWorker : public QObject
{
	Q_OBJECT

private:
	std::vector<cv::Mat> sequence;
	cv::VideoCapture videoCapture;
	//ImageViewer *imageViewer;

signals:
	void ImportingFinished();

public slots:
	Q_INVOKABLE void ImportSequence(const QString& path);

	Q_INVOKABLE void DisplaySequence();

	Q_INVOKABLE void GetFirstFrameROI();
	Q_INVOKABLE void GetSelectedTrackers(std::vector<bool> selectedTrackersQML);
	Q_INVOKABLE void RunAllTrackers();

public:
	void HoughCirclesTracker();
	void DetectMarkersOnImage();
	void RunTrackerCSRT();
	void RunTrackerDaSiamRPN();
	void RunTrackerGOTURN();
};


