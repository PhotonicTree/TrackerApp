#include "BackendWorker.h"

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
	const auto sequencePath = path.toStdString();
	videoCapture.open(sequencePath);

    if (!videoCapture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    cv::Mat frame;
    while (true)
    {
        videoCapture >> frame;
        if (frame.empty())
        {
            break;
        }
        sequence.push_back(frame);
    }
}

Q_INVOKABLE void BackendWorker::DisplaySequence()
{
    ImageViewer* imageViewer(new ImageViewer);

    imageViewer->TriggerStreamingSequence(videoCapture);
}

Q_INVOKABLE cv::VideoCapture BackendWorker::GetCapture()
{
    return videoCapture;
}