#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QObject>
#include <QTimer>
#include <QImage>


/*! \class ImageViewer
	Widget class for displaying cv::Mat images with OpenGL.
*/

class ImageViewer : public QObject
{
	Q_OBJECT
public:

	//! \enum ImageOrientation Image orientation.
	enum class ImageOrientation : int
	{
		Normal = 0,							//!< Normal, no rotation.
		RotateRight = 1,					//!< Rotate Right, 90 degree clockwise.
		Rotate180 = 2,						//!< Rotate 180 degree.
		RotateLeft = 3						//!< Rotate Left, 90 degree counter clockwise.
	};

	/*! Basic constructor.
		\param[in] parent QWidget parent.
	*/
	ImageViewer(QWidget* parent = nullptr);

	//! Basic destructor.
	~ImageViewer();
public:
	void StreamVideo();
	void TriggerStreamingSequence(cv::VideoCapture capture);

public slots:
	void TriggerStreamingPreview(QString path);

private:
	cv::Mat image;												//!< Image to display.
	cv::VideoCapture cap;
	QTimer tUpdate;

signals:
	void newImage(QImage&);
};