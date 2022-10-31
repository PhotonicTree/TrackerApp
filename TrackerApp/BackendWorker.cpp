#include "BackendWorker.h"

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
	const auto sequencePath = path.toStdString();
	videoCapture.open(sequencePath);

    if (!videoCapture.isOpened()) {
        std::cout << "Error opening video stream or file!" << std::endl;
        emit importingFailed();
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
    emit importingFinished();
}

Q_INVOKABLE void BackendWorker::DisplaySequence()
{
    ImageViewer* imageViewer(new ImageViewer);

    imageViewer->TriggerStreamingSequence(videoCapture);
}

Q_INVOKABLE void BackendWorker::GetFirstFrameROIs()
{
    auto firstFrame = sequence.front();
    cv::selectROIs("Select ROI, press Enter and select next. Press ESC to finish selection", firstFrame, ROIs);
    for (const auto& ROI : ROIs)
    {
        cv::rectangle(firstFrame,ROI, cv::Scalar(255, 0, 0), 2, 1);
        cv::circle(firstFrame, cv::Point(ROI.x + ROI.width/2, ROI.y + ROI.height/2), 15, cv::Scalar(0,255,0));
    }
    cv::imshow("SelectedROIs", firstFrame);
    cv::waitKey(0);
    std::cout << "Roi selected";
}

Q_INVOKABLE void BackendWorker::GetSelectedTrackers(std::vector<bool> selectedTrackersQML)
{
    for (auto i = 0; i < selectedTrackersQML.size(); ++i)
    {
        selectedTrackers.insert(std::make_pair(static_cast<BackendWorkerHelpers::TrackerType>(i), selectedTrackersQML[i]));
    }
}

Q_INVOKABLE void BackendWorker::RunAllTrackers()
{
    GetFirstFrameROIs();
    for (const auto selectedTracker : selectedTrackers)
    {
        // Check if current switch is pressed.
        if (selectedTracker.second)
        {
            switch (selectedTracker.first) {
            case BackendWorkerHelpers::TrackerType::CSRT: {
                MultiTrackerCSRT* tracker(new MultiTrackerCSRT);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                        break;
            case BackendWorkerHelpers::TrackerType::MOSSE: {
                MultiTrackerMOSSE* tracker(new MultiTrackerMOSSE);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                         break;
            /*case BackendWorkerHelpers::TrackerType::DaSiamRPN: {
                MultiTrackerMIL* tracker(new MultiTrackerMIL);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                break; */
                /*case BackendWorkerHelpers::TrackerType::GOTURN: {
                    MultiTrackerMIL* tracker(new MultiTrackerMIL);
                    tracker->InitializeTracker(ROIs, sequence);
                    tracker->RunTracking();
                }
                    break; */
            case BackendWorkerHelpers::TrackerType::MIL: {
                MultiTrackerMIL* tracker(new MultiTrackerMIL);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                       break;
            case BackendWorkerHelpers::TrackerType::MedianFlow: {
                MultiTrackerMedianFlow* tracker(new MultiTrackerMedianFlow);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                              break;
            case BackendWorkerHelpers::TrackerType::KCF: {
                MultiTrackerKCF* tracker(new MultiTrackerKCF);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                       break;
            case BackendWorkerHelpers::TrackerType::Boosting: {
                MultiTrackerBoosting* tracker(new MultiTrackerBoosting);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                            break;
            case BackendWorkerHelpers::TrackerType::HoughCircles: {
                MultiHoughCirclesTracker* tracker(new MultiHoughCirclesTracker);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                                break;
            case BackendWorkerHelpers::TrackerType::BlobDetector: {
                MultiBlobDetectorTracker* tracker(new MultiBlobDetectorTracker);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking();
            }
                                                                break;
            }
        }
    }
}