#include "FrameImageProvider.h"

QImage CVMatToQImage(const cv::Mat& inMat)
{
	// according to: https://asmaloney.com/2013/11/code/converting-between-cvmat-and-qimage-or-qpixmap/
	switch (inMat.type())
	{
		// 8-bit, 4 channel
	case CV_8UC4:
	{
		QImage image(inMat.data,
			inMat.cols, inMat.rows,
			static_cast<int>(inMat.step),
			QImage::Format_ARGB32);
		return image.copy();
	}
	// 8-bit, 3 channel
	case CV_8UC3:
	{
		QImage image(inMat.data,
			inMat.cols, inMat.rows,
			static_cast<int>(inMat.step),
			QImage::Format_RGB888);
		return image.rgbSwapped();
	}
	// 8-bit, 1 channel
	case CV_8UC1:
	{
		QImage image(inMat.data,
			inMat.cols, inMat.rows,
			static_cast<int>(inMat.step),
			QImage::Format_Grayscale8);
		return image.copy();
	}
	default:
		std::cout<<(std::to_string(inMat.type()) + " cv::Mat image type not handled in CvMatToQImage");
		break;
	}

	return QImage();
}

FrameImageProvider::FrameImageProvider(QObject* parent) : QObject(parent), QQuickImageProvider(QQuickImageProvider::Image), currentFrameIndex(0)
{
	imageQML = imageQML.scaled(1080, 1920);
}

int FrameImageProvider::GetCurrentFrameIndex() const
{
	return currentFrameIndex;
}

void FrameImageProvider::SetCurrentFrameIndex(int frameIndex)
{
	currentFrameIndex = frameIndex;
}

QImage FrameImageProvider::requestImage(const QString& id, QSize* size, const QSize& requestedSize)
{
	Q_UNUSED(id);

	if (size)
	{
		*size = imageQML.size();
	}

	if (requestedSize.isValid())
	{
		imageQML = imageQML.scaled(requestedSize.width(), requestedSize.height(), Qt::KeepAspectRatio);
	}
	return imageQML;
}

void FrameImageProvider::SetFrameImage(int frameIndex)
{
	auto sequenceLength = sequenceSource->GetSize();
	auto frameIndexToSet = frameIndex >= sequenceLength ? (sequenceLength - 1) : frameIndex < 0 ? 0 : frameIndex;
	std::shared_ptr<SequenceView::Frame> frame = sequenceSource->GetFrame(frameIndexToSet);
	auto frameImage = frame->GetImage();
	currentFrameIndex = frameIndexToSet;
	if (!frameImage.empty())
	{
		// We need 3 channels to draw colorful overlays
		if (frameImage.channels() != 3)
		{
			cv::cvtColor(frameImage, frameImage, cv::COLOR_GRAY2BGR);
		}

		imageQML = CVMatToQImage(frameImage);
	}
	else
	{
		imageQML = QImage();
	}
	updateImage();
}

void FrameImageProvider::updateImage()
{
	if (imageQML.isNull())
	{
		imageQML.load(":/Resources/Images/no_signal_image.jpg");
		imageQML = imageQML.scaled(1080, 1920);
	}

	emit imageChanged();
}
