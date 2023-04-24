#include "BackendWorker.h"
#include "JsonTrackerObject.h"

#include <fstream>

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
    sequence.clear();
	const auto sequencePath = path.toStdString();
	videoCapture.open(sequencePath);
    sequenceName = sequencePath.substr(sequencePath.find_last_of("/\\") + 1);
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
    cv::selectROIs("Select ROI", firstFrame, ROIs);
    cv::destroyWindow("Select ROI");
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
    JsonTrackerObject jsonTrackersObject;
    auto outputFileName = sequenceName.substr(0, sequenceName.find_first_of('.'));
    outputFileName += ".json";
    for (const auto selectedTracker : selectedTrackers)
    {
        // Check if current switch is pressed.
        if (selectedTracker.second)
        {
            switch (selectedTracker.first) {
            case BackendWorkerHelpers::TrackerType::CSRT: {
                MultiTrackerCSRT* tracker(new MultiTrackerCSRT);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
            }
                                                        break;
            case BackendWorkerHelpers::TrackerType::MOSSE: {
                MultiTrackerMOSSE* tracker(new MultiTrackerMOSSE);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
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
                tracker->RunTracking(jsonTrackersObject);
            }
                                                       break;
            case BackendWorkerHelpers::TrackerType::MedianFlow: {
                MultiTrackerMedianFlow* tracker(new MultiTrackerMedianFlow);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
            }
                                                              break;
            case BackendWorkerHelpers::TrackerType::KCF: {
                MultiTrackerKCF* tracker(new MultiTrackerKCF);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
            }
                                                       break;
            case BackendWorkerHelpers::TrackerType::Boosting: {
                MultiTrackerBoosting* tracker(new MultiTrackerBoosting);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
            }
                                                            break;
            case BackendWorkerHelpers::TrackerType::HoughCircles: {
                MultiHoughCirclesTracker* tracker(new MultiHoughCirclesTracker);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
            }
                                                                break;
            case BackendWorkerHelpers::TrackerType::BlobDetector: {
                MultiBlobDetectorTracker* tracker(new MultiBlobDetectorTracker);
                tracker->InitializeTracker(ROIs, sequence);
                tracker->RunTracking(jsonTrackersObject);
            }
                                                                break;
            } 
        }
    }
    std::ofstream exportJSONFile("Results\\" + outputFileName);
    exportJSONFile << jsonTrackersObject.dump(4);
    exportJSONFile.close();
    std::cout << "File exported";
}