#include "BackendWorker.h"
#include "JsonTrackerObject.h"

#include <QUrl>
#include <opencv2/core/utils/filesystem.hpp>

#include <fstream>

namespace
{
/*The QML file dialog hands over a file URL ("file:///..."); OpenCV needs a local path.*/
std::string ToLocalPath(const QString& path)
{
    const QUrl url(path);
    return (url.isLocalFile() ? url.toLocalFile() : path).toStdString();
}

/*Runs one tracker type over the sequence. Automatic storage, so the tracker and its copy of the
    frame list are released as soon as it finishes.*/
template <typename T>
void RunTracker(std::vector<cv::Rect>& ROIs, std::vector<cv::Mat>& sequence, JsonTrackerObject& results)
{
    T tracker;
    tracker.InitializeTracker(ROIs, sequence);
    tracker.RunTracking(results);
}
}   // namespace

Q_INVOKABLE void BackendWorker::ImportSequence(const QString& path)
{
    sequence.clear();
    videoPath = ToLocalPath(path);
    videoCapture.open(videoPath);
    std::string directory = videoPath.substr(0, videoPath.find_last_of("/\\"));
    std::string fileName = videoPath.substr(videoPath.find_last_of("/\\") + 1);
    sequenceName = directory.substr(directory.find_last_of("/\\") + 1) + "/" + fileName;
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

Q_INVOKABLE void BackendWorker::GetSelectedTrackers(std::vector<bool> selectedTrackersQML)
{
    // Replace, not merge: std::map::insert keeps existing keys, so a second run would otherwise
    // reuse the first run's selection.
    selectedTrackers.clear();
    for (size_t i = 0; i < selectedTrackersQML.size(); ++i)
    {
        selectedTrackers[static_cast<BackendWorkerHelpers::TrackerType>(i)] = selectedTrackersQML[i];
    }
}

Q_INVOKABLE void BackendWorker::SetReuseSavedBoxes(bool reuse)
{
    reuseSavedBoxes = reuse;
}

bool BackendWorker::PrepareBoxes()
{
    if (sequence.empty())
    {
        return false;
    }
    const auto boxFile = BoxFileFor(labelDirectory, videoPath);
    std::vector<MarkerBox> boxes;
    if (cv::utils::fs::exists(boxFile))
    {
        boxes = LoadBoxes(boxFile);
    }
    double scale = kDefaultBoxScale;
    if (!(reuseSavedBoxes && !boxes.empty()))
    {
        if (boxes.empty())
        {
            boxes = LabellingProposals(sequence.front(), scale);
        }
        if (LabelBoxes(sequence.front(), boxes, scale, "Label markers - " + sequenceName) != LabelOutcome::Saved)
        {
            std::cout << "Labelling cancelled; nothing was run." << std::endl;
            return false;
        }
        SaveBoxes(boxFile, boxes, videoPath, scale);
        std::cout << "Boxes saved to " << boxFile << std::endl;
    }
    markerBoxes = boxes;
    ROIs = ToRects(boxes);
    return !ROIs.empty();
}

Q_INVOKABLE void BackendWorker::LabelBoxesOnly(const QString& path)
{
    const auto localPath = ToLocalPath(path);
    cv::Mat firstFrame;
    if (!ReadFirstFrame(localPath, firstFrame))
    {
        std::cout << "Could not read the first frame of " << localPath << std::endl;
        return;
    }
    const auto boxFile = BoxFileFor(labelDirectory, localPath);
    std::vector<MarkerBox> boxes;
    if (cv::utils::fs::exists(boxFile))
    {
        boxes = LoadBoxes(boxFile);
    }
    double scale = kDefaultBoxScale;
    if (boxes.empty())
    {
        boxes = LabellingProposals(firstFrame, scale);
    }
    if (LabelBoxes(firstFrame, boxes, scale, "Label markers - " + localPath) == LabelOutcome::Saved)
    {
        SaveBoxes(boxFile, boxes, localPath, scale);
        std::cout << "Saved " << boxes.size() << " boxes to " << boxFile << std::endl;
    }
}

Q_INVOKABLE void BackendWorker::RunAllTrackers()
{
    if (!PrepareBoxes())
    {
        return;
    }
    JsonTrackerObject jsonTrackersObject;
    /*Persist the initial boxes and the build configuration before tracking starts, so a run is
        reproducible and self-documenting even if it is interrupted part way through.*/
    ExportROIs(jsonTrackersObject);
    ExportRunMetadata(jsonTrackersObject);
    for (const auto& selectedTracker : selectedTrackers)
    {
        // Check if current switch is pressed.
        if (!selectedTracker.second)
        {
            continue;
        }
        switch (selectedTracker.first) {
        case BackendWorkerHelpers::TrackerType::CSRT:       RunTracker<MultiTrackerCSRT>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::MOSSE:      RunTracker<MultiTrackerMOSSE>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::DaSiamRPN:  RunTracker<TrackerDaSiamRPNWrapper>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::GOTURN:     RunTracker<TrackerGOTURNWrapper>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::NANO:       RunTracker<TrackerNanoWrapper>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::MIL:        RunTracker<MultiTrackerMIL>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::MedianFlow: RunTracker<MultiTrackerMedianFlow>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::TLD:        RunTracker<MultiTrackerTLD>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::KCF:        RunTracker<MultiTrackerKCF>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::Boosting:   RunTracker<MultiTrackerBoosting>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::HoughCircles: RunTracker<MultiHoughCirclesTracker>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::BlobDetector: {
            MultiBlobDetectorTracker reference;
            reference.InitializeTracker(ROIs, sequence);
            reference.SetLabelledDetection(LabelledBlobParams(markerBoxes), ReferenceRegion(markerBoxes, sequence.front().size()));
            reference.RunTracking(jsonTrackersObject);
            break;
        }
        case BackendWorkerHelpers::TrackerType::VIT:        RunTracker<TrackerVitWrapper>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::Hybrid:     RunTracker<HybridTracker>(ROIs, sequence, jsonTrackersObject); break;
        case BackendWorkerHelpers::TrackerType::Reference: {
            MultiBlobDetectorTrackerReference reference;
            reference.InitializeTracker(ROIs, sequence);
            reference.SetLabelledDetection(LabelledBlobParams(markerBoxes), ReferenceRegion(markerBoxes, sequence.front().size()));
            reference.RunTracking(jsonTrackersObject);
            break;
        }
        }
    }
    const std::string outputPath = "Results/" + sequenceName.substr(0, sequenceName.find_last_of('.')) + ".json";
    cv::utils::fs::createDirectories(outputPath.substr(0, outputPath.find_last_of('/')));
    std::ofstream exportJSONFile(outputPath);
    if (!exportJSONFile)
    {
        std::cout << "Could not write " << outputPath << std::endl;
        return;
    }
    exportJSONFile << jsonTrackersObject.dump(4);
    exportJSONFile.close();
    std::cout << "File exported to " << outputPath << std::endl;
}

void BackendWorker::ExportROIs(JsonTrackerObject& jsonObject) const
{
    /*RunAllTrackers selects the ROIs once and hands the same vector to every tracker, so all
        trackers start from identical boxes. Recording them makes that verifiable and lets a run
        be reproduced exactly.*/
    for (size_t i = 0; i < markerBoxes.size(); ++i)
    {
        const auto& marker = markerBoxes[i];
        jsonObject["_rois"][std::to_string(i)] = {
            {"x", marker.box.x}, {"y", marker.box.y}, {"w", marker.box.width}, {"h", marker.box.height},
            {"cx", marker.centre.x}, {"cy", marker.centre.y}, {"d", marker.diameter},
            {"adjusted", marker.adjusted} };
    }
}

void BackendWorker::ExportRunMetadata(JsonTrackerObject& jsonObject) const
{
    jsonObject["_meta"]["opencv_version"] = CV_VERSION;
    jsonObject["_meta"]["cv_num_threads"] = cv::getNumThreads();
    jsonObject["_meta"]["cpu_features"] = cv::getCPUFeaturesLine();
    jsonObject["_meta"]["sequence"] = sequenceName;
    jsonObject["_meta"]["frames"] = static_cast<int>(sequence.size());
    jsonObject["_meta"]["n_targets"] = static_cast<int>(markerBoxes.size());
    jsonObject["_meta"]["roi_source"] = "file: " + BoxFileFor(labelDirectory, videoPath);
    jsonObject["_meta"]["coordinate_precision"] = RunConfig().coordinatePrecision;
}
