#include "FrameImageProvider.h"


FrameImageProvider::FrameImageProvider(QObject* parent) : QObject(parent), QQuickImageProvider(QQuickImageProvider::Image), currentFrameIndex(0)
{
	imageQML.load(":/Resources/Images/no_signal_image.jpg");
	imageQML = imageQML.scaled(1080, 1920);
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


void FrameImageProvider::updateImage()
{
	emit imageChanged();
}
