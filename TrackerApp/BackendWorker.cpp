#include "BackendWorker.h"

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
	const auto sequencePath = path.toStdString();
	cv::VideoCapture capture(sequencePath);

    if (!capture.isOpened()) {
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    cv::Mat frame;
    while (true)
    {
        capture >> frame;
        if (frame.empty())
        {
            break;
        }
        sequence.push_back(frame);
    }
}

Q_INVOKABLE void BackendWorker::DisplaySequence(const QString& path)
{
    return Q_INVOKABLE void();
}
