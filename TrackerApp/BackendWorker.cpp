#include "BackendWorker.h"

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
	const auto sequencePath = path.toStdString();
	videoCapture.open(sequencePath);

    if (!videoCapture.isOpened()) {
        std::cout << "Error opening video stream or file!" << std::endl;
        return;
    }

    while (videoCapture.isOpened())
    {
        cv::Mat frame;
        videoCapture >> frame;
        if (frame.empty())
        {
            break;
        }
        sequence.push_back(frame);
    }
    videoCapture.release();
    ImportingFinished();
}

Q_INVOKABLE void BackendWorker::DisplaySequence()
{
    ImageViewer* imageViewer(new ImageViewer);

    imageViewer->TriggerStreamingSequence(videoCapture);
}

Q_INVOKABLE void BackendWorker::GetFirstFrameROIs()
{
    auto& firstFrame = sequence.front();
    cv::selectROIs("tracker", firstFrame, ROIs);
    std::cout << "Roi selected";
}

Q_INVOKABLE void BackendWorker::GetSelectedTrackers(std::vector<bool> selectedTrackersQML)
{
    for (auto i = 0; i < selectedTrackersQML.size(); ++i)
    {
        selectedTrackers.insert(std::make_pair(static_cast<BackendWorkerHelpers::TrackerType>(i), selectedTrackersQML[i]));
    }
    RunAllTrackers();
}

Q_INVOKABLE void BackendWorker::RunAllTrackers()
{ 
    for (const auto selectedTracker : selectedTrackers)
    {
        // Check if current switch is pressed.
        if (selectedTracker.second)
        {
            switch (selectedTracker.first) {
            case BackendWorkerHelpers::TrackerType::CSRT: RunTrackerCSRT();
                break;
            case BackendWorkerHelpers::TrackerType::MOSSE: RunTrackerMOSSE();
                break;
            case BackendWorkerHelpers::TrackerType::DaSiamRPN: RunTrackerDaSiamRPN();
                break;
            case BackendWorkerHelpers::TrackerType::GOTURN: RunTrackerGOTURN();
                break;
                //case BackendWorkerHelpers::TrackerType::CSRT: RunTrackerCSRT();
                //    break;
                //case BackendWorkerHelpers::TrackerType::CSRT: RunTrackerCSRT();
                //    break;
            }
        }
    }
}