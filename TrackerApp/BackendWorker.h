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
		BlobDetector,
		MOSSE,
		CSRT,
		MIL,
		TLD,
		Boosting,
		KCF,
		MedianFlow,
		GOTURN,
		DaSiamRPN,
		NANO,
		VIT,
		Hybrid,
		HoughCircles,
		Reference
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

	/*Opens frame 0 of a video in the labelling window and saves the boxes to its box file,
		without running any tracker. Used to prepare videos for scripted runs.
		\param[in] path QString with path (or file URL) to the video.
	*/
	Q_INVOKABLE void LabelBoxesOnly(const QString& path);

	/*When set, a video that already has a box file is run from it without opening the
		labelling window.
		\param[in] reuse true to skip the review of existing boxes.
	*/
	Q_INVOKABLE void SetReuseSavedBoxes(bool reuse);

private:
	/*Loads the boxes for the current video, lets the operator review or label them, and saves
		them to the box file. The boxes are always saved, so every run from the GUI can later be
		repeated with the headless harness.
		\returns false if the operator cancelled or no boxes are available.
	*/
	bool PrepareBoxes();

	/*Writes the initial ROIs into the results object, so the boxes every tracker started from
		are recorded with the results.
		\param[in,out] jsonObject Results object receiving the "_rois" section.
	*/
	void ExportROIs(JsonTrackerObject& jsonObject) const;

	/*Writes build and run metadata so each results file documents the configuration it came from.
		\param[in,out] jsonObject Results object receiving the "_meta" section.
	*/
	void ExportRunMetadata(JsonTrackerObject& jsonObject) const;

	std::vector<cv::Mat> sequence;		// Vector with sequence of images to be analyzed.
	cv::VideoCapture videoCapture;		// VideoCapture loaded by user.
	std::vector<cv::Rect> ROIs;			// Vector of cv::Rect with selected regions of interests by user.
	std::vector<MarkerBox> markerBoxes;	// The same boxes with the marker centres and diameters.
	std::string videoPath;				// Local path of the loaded video.
	std::string labelDirectory = "Labels";	// Root directory of the box files.
	bool reuseSavedBoxes = false;		// Run from an existing box file without review.
	std::map<BackendWorkerHelpers::TrackerType, bool> selectedTrackers;		// Map with tracker types as a key and them boolean state as value.
	std::string sequenceName;
};


