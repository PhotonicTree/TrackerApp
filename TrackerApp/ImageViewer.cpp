#include "ImageViewer.h"

ImageViewer::ImageViewer(QWidget* parent)
{
	qRegisterMetaType<cv::Mat>("cv::Mat");
	qRegisterMetaType<ImageViewer::ImageOrientation>("ImageViewer::ImageOrientation");
}

ImageViewer::~ImageViewer()
{
}