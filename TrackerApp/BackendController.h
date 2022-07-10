#pragma once
#include "BackendWorker.h"
#include <QThread>

/*!	\file BackendController.h
	This file contains BackendController class for creating all connections with QML frontend and heavy operations on another thread.
*/

class BackendController : public QObject
{
	Q_OBJECT
public:
	/*! BackendController constructor.
		\param[in] backendWorker Backend worker.
		\param[in] parent Parent QObject.
	*/
	BackendController(BackendWorker* backendWorker, QObject* parent = nullptr);

	//! Destructor.
	~BackendController();

signals:
	//! Signal used to emit to start importing action.
	void startImporting(const QString& path);

	void startDisplaying();

	void importingFinished();

private:
	BackendWorker* backendWorker; //!< A pointer to backend Worker, which is created in main file and do all backend methods.
	QThread backendThread;		  //!< QThread for heavy operations, which runs simultaneously and is called from QML by connects.

private:
	cv::Mat image;												//!< Image to display.
	cv::VideoCapture cap;
	//QTimer tUpdate;

signals:
	void newImage(QImage&);

public:
	//void streamVideo();
};
