#include "ImageViewer.h"

ImageViewer::ImageViewer(QWidget* parent)
{
	qRegisterMetaType<cv::Mat>("cv::Mat");
	qRegisterMetaType<ImageViewer::ImageOrientation>("ImageViewer::ImageOrientation");
	connect(&tUpdate, &QTimer::timeout, this, &ImageViewer::StreamVideo);
}

ImageViewer::~ImageViewer()
{
	cap.release();
	tUpdate.stop();
}

void ImageViewer::StreamVideo()
{
	cap >> image;
	QImage img = QImage(image.data, image.cols, image.rows, QImage::Format_RGB888).rgbSwapped();
	emit newImage(img);
}

void ImageViewer::TriggerStreamingPreview(QString path)
{
	if (path.length() == 1)
		cap.open(path.toInt());
	else
		cap.open(path.toStdString());

	double fps = cap.get(cv::CAP_PROP_FPS);
	tUpdate.start(1000 / fps);
}

void ImageViewer::TriggerStreamingSequence(cv::VideoCapture capture)
{
	if (!capture.isOpened())
	{
		double fps = cap.get(cv::CAP_PROP_FPS);
		tUpdate.start(1000 / fps);
	}
	else
	{
		std::cout << "Couldn't display video capture." << std::endl;
	}
}