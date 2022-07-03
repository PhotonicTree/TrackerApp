#include "ImageViewer.h"

ImageViewer::ImageViewer(QWidget* parent)
{
	qRegisterMetaType<cv::Mat>("cv::Mat");
	qRegisterMetaType<ImageViewer::ImageOrientation>("ImageViewer::ImageOrientation");
	connect(&tUpdate, &QTimer::timeout, this, &ImageViewer::streamVideo);
}

ImageViewer::~ImageViewer()
{
	cap.release();
	tUpdate.stop();
}

void ImageViewer::streamVideo()
{
	cap >> image;
	QImage img = QImage(image.data, image.cols, image.rows, QImage::Format_RGB888).rgbSwapped();
	emit newImage(img);
}

void ImageViewer::openVideoCamera(QString path)
{
	if (path.length() == 1)
		cap.open(path.toInt());
	else
		cap.open(path.toStdString());

	double fps = cap.get(cv::CAP_PROP_FPS);
	tUpdate.start(1000 / fps);
}