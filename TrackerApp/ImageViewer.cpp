#include "ImageViewer.h"

ImageViewer::ImageViewer(QWidget* parent)
{
	qRegisterMetaType<cv::Mat>("cv::Mat");
	qRegisterMetaType<ImageViewer::ImageOrientation>("ImageViewer::ImageOrientation");
	connect(&timeUpdate, &QTimer::timeout, this, &ImageViewer::StreamVideo);
}

ImageViewer::~ImageViewer()
{
	videoCapture.release();
	timeUpdate.stop();
}

void ImageViewer::StreamVideo()
{
	videoCapture >> image;
	QImage img = QImage(image.data, image.cols, image.rows, QImage::Format_RGB888).rgbSwapped();
	emit newImage(img);
}

void ImageViewer::TriggerStreamingPreview(QString path)
{
	if (path.length() == 1)
		videoCapture.open(path.toInt());
	else
		videoCapture.open(path.toStdString());

	double fps = videoCapture.get(cv::CAP_PROP_FPS);
	timeUpdate.start(1000 / fps);
}

void ImageViewer::TriggerStreamingSequence(cv::VideoCapture capture)
{
	if (!capture.isOpened())
	{
		double fps = videoCapture.get(cv::CAP_PROP_FPS);
		timeUpdate.start(1000 / fps);
	}
	else
	{
		std::cout << "Couldn't display video videoCapture." << std::endl;
	}
}