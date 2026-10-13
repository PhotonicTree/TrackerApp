#include "Trackers.h"
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <stdexcept>

#include <opencv2/dnn.hpp>

#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif

/*typeid(T).name() is implementation-defined. MSVC yields "class cv::legacy::TrackerCSRT",
    but the Itanium ABI used by GCC/Clang yields the mangled "N2cv6legacy11TrackerCSRTE",
    which contains no ':' and so survives the trim below unchanged. Demangle first so the
    keys written to the output JSON are the same bare class names on every platform.*/
static std::string GetBareTrackerName(const std::type_info& trackerType)
{
    std::string name = trackerType.name();
#if defined(__GNUC__) || defined(__clang__)
    int status = 0;
    char* demangled = abi::__cxa_demangle(trackerType.name(), nullptr, nullptr, &status);
    if (status == 0 && demangled != nullptr)
    {
        name = demangled;
    }
    std::free(demangled);
#endif
    name.erase(0, name.find_last_of(':') + 1);
    return name;
}

namespace
{
/*Centre and size of a tracker box. Rect2d keeps the legacy trackers' fractional coordinates; the
    integer cv::Rect of the modern trackers converts exactly, and computing the centre in double
    keeps the half pixel that integer division by 2 would lose for an odd width or height.*/
TrackedTarget ToTarget(const cv::Rect2d& box, bool valid)
{
    TrackedTarget target;
    target.centreX = box.x + box.width * 0.5;
    target.centreY = box.y + box.height * 0.5;
    target.width = box.width;
    target.height = box.height;
    target.valid = valid;
    return target;
}

/*Serialises one frame as "(cx;cy;w;h),(cx;cy;w;h),..." - the format the analysis scripts
    expect - with sub-pixel values.

    Separators are placed by index rather than by comparing values, so a coordinate that happens
    to equal another field cannot close a bracket early.*/
std::string BuildFrameLine(const std::vector<TrackedTarget>& targets)
{
    std::string line;
    for (size_t k = 0; k < targets.size(); ++k)
    {
        const auto& target = targets[k];
        line += "(" + FormatCoordinate(target.centreX) + ";" + FormatCoordinate(target.centreY)
              + ";" + FormatCoordinate(target.width)   + ";" + FormatCoordinate(target.height) + ")";
        if (k + 1 < targets.size())
        {
            line += ",";
        }
    }
    return line;
}

/*Zero-padded frame key, matching the existing "0001" convention.*/
std::string BuildFrameKey(size_t frameIndex)
{
    const size_t width = 4;
    auto frameNumber = std::to_string(frameIndex);
    const size_t padding = width - std::min(width, frameNumber.size());
    frameNumber.insert(0, padding, '0');
    return frameNumber;
}

/*Writes the per-run diagnostics: per-frame timings (from which latency means and standard
    deviations are computed), the number of targets actually timed, and the per-frame validity
    flags that the OpenCV APIs report.

    These live under a separate "_diagnostics" object so the per-tracker frame keys keep their
    format and existing consumers are unaffected.*/
void WriteDiagnostics(JsonTrackerObject& jsonTrackersObject, const std::string& trackerTypeName,
    const std::vector<long long>& frameTimesUs, size_t targetCount,
    const nlohmann::json& validity, double framesPerSecond)
{
    auto& diagnostics = jsonTrackersObject["_diagnostics"][trackerTypeName];
    diagnostics["frame_times_us"] = frameTimesUs;
    diagnostics["n_targets"] = static_cast<int>(targetCount);
    diagnostics["valid"] = validity;
    diagnostics["fps"] = framesPerSecond;
}

/*Frames actually timed. The loop starts at index 1 (frame 0 initialises the trackers), so this
    is one fewer than the sequence length.*/
double TimedFrameCount(size_t sequenceSize)
{
    return sequenceSize > 1 ? static_cast<double>(sequenceSize - 1) : 1.0;
}
}   // namespace

TrackerRunConfig& RunConfig()
{
    static TrackerRunConfig configuration;
    return configuration;
}

std::string FormatCoordinate(double value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(RunConfig().coordinatePrecision) << value;
    return stream.str();
}

BaseTracker::BaseTracker()
{
}

BaseTracker::~BaseTracker()
{
}

/*Runs one of the opencv_contrib legacy trackers over the whole sequence.

    cv::legacy::MultiTracker holds every ROI and advances all of them in a single update(),
    so the timed region covers all targets. Note that the legacy API reports one aggregate
    success flag for the whole set, not one per target; the flag is recorded per frame and
    replicated across targets, which is documented in the results file.*/
template <typename T> void RunTrackingForOpenCVLegacyTracker(std::vector<cv::Rect> ROIs, std::vector<cv::Mat> sequence, JsonTrackerObject &jsonTrackersObject)
{
    /*One tracker instance per ROI rather than cv::legacy::MultiTracker.

        MultiTracker::update is a loop over exactly these instances combining their results with
        status &= ..., so the geometry and the timed work are identical - but it returns a single
        flag ANDed across every target. One low-confidence marker then marks the whole frame
        invalid, which makes per-target validity unusable and defeats the failure-timing metrics.
        Holding the instances directly gives each target its own flag.*/
    std::vector<cv::Ptr<cv::legacy::Tracker>> trackers;
    std::vector<cv::Rect2d> trackerObjects;
    trackers.reserve(ROIs.size());
    trackerObjects.reserve(ROIs.size());
    for (const auto& ROI : ROIs)
    {
        auto tracker = T::create();
        tracker->init(sequence.front(), cv::Rect2d(ROI));
        trackers.push_back(tracker);
        trackerObjects.push_back(cv::Rect2d(ROI));
    }
    std::vector<char> updateSucceeded(trackers.size(), 0);
    // Get name of current tracker and push to json file.
    const std::string trackerTypeName = GetBareTrackerName(typeid(T));

    long long timeCount = 0;
    std::vector<long long> frameTimesUs;
    frameTimesUs.reserve(sequence.size());
    nlohmann::json validityPerFrame = nlohmann::json::object();

    for (size_t i = 1; i < sequence.size(); i++)
    {
        auto image = sequence.at(i);
        cv::Mat show;
        if (RunConfig().showWindows)
        {
            show = image.clone();
        }

        const auto tik = std::chrono::steady_clock::now();
        for (size_t k = 0; k < trackers.size(); ++k)
        {
            updateSucceeded[k] = trackers[k]->update(image, trackerObjects[k]) ? 1 : 0;
        }
        const auto frameTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - tik).count();
        frameTimesUs.push_back(frameTimeUs);
        timeCount += frameTimeUs;

        std::vector<TrackedTarget> targets;
        targets.reserve(trackerObjects.size());
        std::vector<bool> validity;
        validity.reserve(trackerObjects.size());
        for (size_t k = 0; k < trackerObjects.size(); ++k)
        {
            const auto& trackerObject = trackerObjects[k];
            if (RunConfig().showWindows)
            {
                cv::rectangle(show, trackerObject, cv::Scalar(255, 0, 0), 2, 1);
            }
            /*The legacy trackers report cv::Rect2d, so these coordinates are fractional and are
                kept as such. Truncating them to integers would shift every centre by about half
                a pixel per axis relative to the sub-pixel reference.*/
            const auto target = ToTarget(trackerObject, updateSucceeded[k] != 0);
            targets.push_back(target);
            validity.push_back(target.valid);
        }

        const auto frameNumber = BuildFrameKey(i);
        jsonTrackersObject[trackerTypeName][frameNumber] = BuildFrameLine(targets);
        validityPerFrame[frameNumber] = validity;
        if (RunConfig().logEveryFrame)
        {
            std::cout << trackerTypeName << " - " << frameNumber << std::endl;
        }
        if (RunConfig().showWindows)
        {
            cv::imshow("Tracker result. Press 'x' to quit", show);
            if (cv::waitKey(1) == 27) break;
        }
    }

    const auto durationInSeconds = timeCount / 1000000.0;
    std::cout << trackerTypeName << " time " << durationInSeconds << std::endl;
    const auto fps = TimedFrameCount(sequence.size()) / durationInSeconds;
    jsonTrackersObject[trackerTypeName]["FPS"] = std::to_string(fps);
    WriteDiagnostics(jsonTrackersObject, trackerTypeName, frameTimesUs, ROIs.size(), validityPerFrame, fps);
}

/*Runs one of the modern (cv::Tracker) implementations over the whole sequence.

    One tracker instance is created per ROI and all of them are advanced inside the timed
    region, so this path measures the same thing as the legacy path for any number of targets.*/
template <typename T> void RunTrackingForOpenCVTracker(std::vector<cv::Rect> ROIs, std::vector<cv::Mat> sequence, JsonTrackerObject& jsonTrackersObject)
{
    std::vector<cv::Ptr<T>> trackers;
    trackers.reserve(ROIs.size());
    for (const auto& ROI : ROIs)
    {
        auto tracker = T::create();
        tracker->init(sequence.front(), ROI);
        trackers.push_back(tracker);
    }

    // Get name of current tracker and push to json file.
    const std::string trackerTypeName = GetBareTrackerName(typeid(T));

    long long timeCount = 0;
    std::vector<long long> frameTimesUs;
    frameTimesUs.reserve(sequence.size());
    nlohmann::json validityPerFrame = nlohmann::json::object();

    std::vector<cv::Rect> currentRectangles(trackers.size());
    std::vector<char> updateSucceeded(trackers.size(), 0);

    for (size_t i = 1; i < sequence.size(); i++)
    {
        auto image = sequence.at(i);
        cv::Mat show;
        if (RunConfig().showWindows)
        {
            show = image.clone();
        }

        const auto tik = std::chrono::steady_clock::now();
        for (size_t k = 0; k < trackers.size(); ++k)
        {
            updateSucceeded[k] = trackers[k]->update(image, currentRectangles[k]) ? 1 : 0;
        }
        const auto frameTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - tik).count();
        frameTimesUs.push_back(frameTimeUs);
        timeCount += frameTimeUs;

        std::vector<TrackedTarget> targets;
        targets.reserve(trackers.size());
        std::vector<bool> validity;
        validity.reserve(trackers.size());
        for (size_t k = 0; k < trackers.size(); ++k)
        {
            const auto& currentRectangle = currentRectangles[k];
            if (RunConfig().showWindows)
            {
                cv::rectangle(show, currentRectangle, cv::Scalar(255, 0, 0), 2, 1);
            }
            // cv::Tracker::update yields an integer cv::Rect: the corner stays quantised.
            const auto target = ToTarget(cv::Rect2d(currentRectangle), updateSucceeded[k] != 0);
            targets.push_back(target);
            validity.push_back(target.valid);
        }

        const auto frameNumber = BuildFrameKey(i);
        jsonTrackersObject[trackerTypeName][frameNumber] = BuildFrameLine(targets);
        validityPerFrame[frameNumber] = validity;
        if (RunConfig().logEveryFrame)
        {
            std::cout << trackerTypeName << " - " << frameNumber << std::endl;
        }
        if (RunConfig().showWindows)
        {
            cv::imshow("Tracker result. Press 'x' to quit", show);
            if (cv::waitKey(1) == 27) break;
        }
    }

    const auto durationInSeconds = timeCount / 1000000.0;
    std::cout << trackerTypeName << " time " << durationInSeconds << std::endl;
    const auto fps = TimedFrameCount(sequence.size()) / durationInSeconds;
    jsonTrackersObject[trackerTypeName]["FPS"] = std::to_string(fps);
    WriteDiagnostics(jsonTrackersObject, trackerTypeName, frameTimesUs, ROIs.size(), validityPerFrame, fps);
}

void MultiTrackerMIL::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMIL>(ROIs, sequence, jsonObject);
}

void MultiTrackerMIL::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{ 
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMOSSE::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMOSSE::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMOSSE>(ROIs, sequence, jsonObject);
}

void MultiTrackerCSRT::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerCSRT::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerCSRT>(ROIs, sequence, jsonObject);
}

void TrackerGOTURNWrapper::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void TrackerGOTURNWrapper::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVTracker<cv::TrackerGOTURN>(ROIs, sequence, jsonObject);
}

void TrackerDaSiamRPNWrapper::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void TrackerDaSiamRPNWrapper::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVTracker<cv::TrackerDaSiamRPN>(ROIs, sequence, jsonObject);
}

void TrackerNanoWrapper::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void TrackerNanoWrapper::RunTracking(JsonTrackerObject& jsonObject)
{
    RunTrackingForOpenCVTracker<cv::TrackerNano>(ROIs, sequence, jsonObject);
}

void TrackerVitWrapper::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void TrackerVitWrapper::RunTracking(JsonTrackerObject& jsonObject)
{
    RunTrackingForOpenCVTracker<cv::TrackerVit>(ROIs, sequence, jsonObject);
}

void HybridTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    InitializeTracker(boundingBoxes, images, Parameters());
}

void HybridTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images,
    const Parameters& parameters)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->hybridParameters = parameters;
}

namespace
{
const char* const kHybridSection = "HybridVitCsrt";

/*One marker of the hybrid pipeline: ViT is the primary state-estimator and CSRT the fallback
    state-corrector. Both are advanced on every frame. While the two agree - their centres lie
    within about one marker diameter of each other - the primary's box is taken, as it is the more
    precise. When the primary reports failure or disagrees with the fallback, the fallback's box is
    taken for that frame. The primary is re-initialised from the fallback's box when it reports
    failure or after Parameters::reinitialiseAfterFrames consecutive frames of disagreement: a
    re-initialised primary learns its template from the fallback's box and so inherits the
    fallback's centring error, whereas after a brief disagreement it keeps its own template and can
    come back into agreement.*/
class HybridTarget
{
public:
    HybridTarget(const cv::Mat& firstFrame, const cv::Rect& roi)
        : primary(cv::TrackerVit::create()),
          fallback(cv::legacy::TrackerCSRT::create()),
          initialShortSide(static_cast<double>(std::min(roi.width, roi.height)))
    {
        primary->init(firstFrame, roi);
        fallback->init(firstFrame, cv::Rect2d(roi));
    }

    /*Advances both trackers on one frame; usedFallback reports whether the fallback's box was taken.*/
    TrackedTarget Update(const cv::Mat& image, const HybridTracker::Parameters& parameters, bool& usedFallback)
    {
        cv::Rect primaryBox;
        const bool primaryOk = primary->update(image, primaryBox);

        cv::Rect2d fallbackBox;
        const bool fallbackOk = fallback->update(image, fallbackBox);

        /*A primary that slides onto a neighbouring marker or a skin fold leaves the fallback
            behind; when both follow fast facial motion, they stay together.*/
        const cv::Point2d primaryCentre(primaryBox.x + primaryBox.width * 0.5,
                                        primaryBox.y + primaryBox.height * 0.5);
        const cv::Point2d fallbackCentre(fallbackBox.x + fallbackBox.width * 0.5,
                                         fallbackBox.y + fallbackBox.height * 0.5);
        const bool disagrees = fallbackOk && cv::norm(primaryCentre - fallbackCentre)
                                             > parameters.maxDisagreementBoxFraction * initialShortSide;

        TrackedTarget target;
        usedFallback = (!primaryOk || disagrees) && fallbackOk;
        disagreeingFrames = disagrees ? disagreeingFrames + 1 : 0;
        if (usedFallback)
        {
            target = ToTarget(fallbackBox, true);
            if (!primaryOk || disagreeingFrames >= parameters.reinitialiseAfterFrames)
            {
                primary = cv::TrackerVit::create();
                primary->init(image, cv::Rect(fallbackBox));
                disagreeingFrames = 0;
            }
        }
        else
        {
            target = ToTarget(cv::Rect2d(primaryBox), primaryOk);
        }
        return target;
    }

private:
    cv::Ptr<cv::TrackerVit> primary;
    cv::Ptr<cv::legacy::TrackerCSRT> fallback;
    double initialShortSide;
    int disagreeingFrames = 0;      // consecutive frames in which the two trackers disagreed
};
}   // namespace

void HybridTracker::RunTracking(JsonTrackerObject& jsonObject)
{
    /*Both trackers run inside the timed region, so the reported cost is the true cost of the
        hybrid rather than of ViT alone.*/
    const std::string trackerTypeName = kHybridSection;

    std::vector<HybridTarget> hybridTargets;
    hybridTargets.reserve(ROIs.size());
    for (const auto& ROI : ROIs)
    {
        hybridTargets.emplace_back(sequence.front(), ROI);
    }

    long long timeCount = 0;
    long long switchCount = 0;
    std::vector<long long> frameTimesUs;
    frameTimesUs.reserve(sequence.size());
    nlohmann::json validityPerFrame = nlohmann::json::object();
    nlohmann::json switchesPerFrame = nlohmann::json::object();

    for (size_t i = 1; i < sequence.size(); i++)
    {
        auto image = sequence.at(i);
        cv::Mat show;
        if (RunConfig().showWindows)
        {
            show = image.clone();
        }

        std::vector<TrackedTarget> targets;
        std::vector<bool> validity;
        std::vector<bool> switched;
        targets.reserve(hybridTargets.size());

        const auto tik = std::chrono::steady_clock::now();
        for (auto& hybridTarget : hybridTargets)
        {
            bool usedFallback = false;
            const auto target = hybridTarget.Update(image, hybridParameters, usedFallback);
            switchCount += usedFallback ? 1 : 0;
            targets.push_back(target);
            validity.push_back(target.valid);
            switched.push_back(usedFallback);
        }
        const auto frameTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - tik).count();
        frameTimesUs.push_back(frameTimeUs);
        timeCount += frameTimeUs;

        if (RunConfig().showWindows)
        {
            for (size_t k = 0; k < targets.size(); ++k)
            {
                const cv::Rect box(cvRound(targets[k].centreX - targets[k].width * 0.5),
                                   cvRound(targets[k].centreY - targets[k].height * 0.5),
                                   cvRound(targets[k].width), cvRound(targets[k].height));
                // Red marks a frame the fallback had to correct, blue a frame the primary held.
                cv::rectangle(show, box, switched[k] ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 0, 0), 2, 1);
            }
        }

        const auto frameNumber = BuildFrameKey(i);
        jsonObject[trackerTypeName][frameNumber] = BuildFrameLine(targets);
        validityPerFrame[frameNumber] = validity;
        switchesPerFrame[frameNumber] = switched;
        if (RunConfig().logEveryFrame)
        {
            std::cout << trackerTypeName << " - " << frameNumber << std::endl;
        }
        if (RunConfig().showWindows)
        {
            cv::imshow("Tracker result. Press 'x' to quit", show);
            if (cv::waitKey(1) == 27) break;
        }
    }

    const auto durationInSeconds = timeCount / 1000000.0;
    std::cout << trackerTypeName << " time " << durationInSeconds
              << " (fallback corrections: " << switchCount << ")" << std::endl;
    const auto fps = TimedFrameCount(sequence.size()) / durationInSeconds;
    jsonObject[trackerTypeName]["FPS"] = std::to_string(fps);
    WriteDiagnostics(jsonObject, trackerTypeName, frameTimesUs, ROIs.size(), validityPerFrame, fps);
    jsonObject["_diagnostics"][trackerTypeName]["fallback_switches"] = switchesPerFrame;
    jsonObject["_diagnostics"][trackerTypeName]["fallback_switch_count"] = switchCount;
    jsonObject["_diagnostics"][trackerTypeName]["primary"] = "TrackerVit";
    jsonObject["_diagnostics"][trackerTypeName]["fallback"] = "TrackerCSRT";
}

void MultiTrackerTLD::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerTLD::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerTLD>(ROIs, sequence, jsonObject);
}

void MultiTrackerBoosting::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerBoosting::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerBoosting>(ROIs, sequence, jsonObject);
}

void MultiTrackerKCF::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerKCF::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerKCF>(ROIs, sequence, jsonObject);
}

void MultiTrackerMedianFlow::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiTrackerMedianFlow::RunTracking(JsonTrackerObject &jsonObject)
{
    RunTrackingForOpenCVLegacyTracker<cv::legacy::TrackerMedianFlow>(ROIs, sequence, jsonObject);
}

void MultiHoughCirclesTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
}

void MultiHoughCirclesTracker::RunTracking(JsonTrackerObject &jsonObject)
{
    for (const auto& frame : sequence)
    {
        for (const auto& ROI : ROIs)
        {
            cv::Mat3b frame3b = frame.clone();
            std::vector<cv::Vec3f> circles;
            cv::Mat grayImage;
            auto ROIImage = frame3b(ROI);
            cv::cvtColor(ROIImage, grayImage, cv::COLOR_BGR2GRAY);
            cv::medianBlur(grayImage, grayImage, 5);
            cv::HoughCircles(grayImage, circles, cv::HOUGH_GRADIENT, 1, grayImage.rows / 16, 140, 10, 1, 10);

            for (size_t i = 0; i < circles.size(); i++)
            {
                cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
                int radius = cvRound(circles[i][2]);
                // draw the circle center
                cv::circle(ROIImage, center, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                // draw the circle outline
                circle(ROIImage, center, radius, cv::Scalar(0, 0, 255), 3, 8, 0);
            }
            cv::imshow("HoughCircles Result", frame3b);
            if (cv::waitKey(100) == 27) break;
        }
    }
}

void MultiBlobDetectorTracker::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius,
    MarkerPreset preset)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->circleRadius = radius;
    this->markerPreset = preset;
    circleBlobDetectorParams = BlobParams(preset);
}

namespace
{
/*Records which detector settings and search region produced a reference stream.*/
void DescribeReference(JsonTrackerObject& jsonObject, const std::string& sectionName, bool labelled,
    MarkerPreset preset, const cv::SimpleBlobDetector::Params& params, const cv::Rect& region)
{
    jsonObject["_meta"]["presets"][sectionName] = labelled ? std::string("labelled") : PresetName(preset);
    auto& detector = jsonObject["_meta"]["reference_detector"][sectionName];
    detector["min_area"] = params.minArea;
    detector["max_area"] = params.maxArea;
    detector["min_circularity"] = params.minCircularity;
    detector["min_convexity"] = params.minConvexity;
    detector["min_inertia_ratio"] = params.minInertiaRatio;
    detector["thresholds"] = { params.minThreshold, params.maxThreshold, params.thresholdStep };
    detector["region"] = { region.x, region.y, region.width, region.height };
}

cv::Rect SearchRegion(const cv::Rect& requested, const cv::Mat& image)
{
    const cv::Rect frame(0, 0, image.cols, image.rows);
    return requested.area() > 0 ? (requested & frame) : frame;
}

/*Blob Detection in the search region of one frame, with keypoints in frame coordinates.*/
std::vector<cv::KeyPoint> DetectBlobs(cv::SimpleBlobDetector& detector, const cv::Mat& image, const cv::Rect& region)
{
    std::vector<cv::KeyPoint> keypoints;
    detector.detect(image(region), keypoints);
    for (auto& keypoint : keypoints)
    {
        keypoint.pt += cv::Point2f(static_cast<float>(region.x), static_cast<float>(region.y));
    }
    return keypoints;
}

/*One frame of the Blob Detection stream: "(x;y),(x;y),", or "(0;0)" when nothing was found.

    SimpleBlobDetector already produces a sub-pixel centre, and it is written as such: this
    stream is the reference every tracker is measured against.*/
std::string BlobFrameLine(const std::vector<cv::KeyPoint>& keypoints)
{
    std::string line;
    for (const auto& keypoint : keypoints)
    {
        line += "(" + FormatCoordinate(keypoint.pt.x) + ";" + FormatCoordinate(keypoint.pt.y) + "),";
    }
    return keypoints.empty() ? std::string("(0;0)") : line;
}

/*One frame of the PCWE stream: refined centres in frame coordinates, or "(0;0)" when none.*/
std::string PcweFrameLine(const WeightedEdgesExtractor::Parameters& parameters, const cv::Mat& image, const cv::Rect& region)
{
    WeightedEdgesExtractor weightedEdgesExtractor(parameters);
    const auto refinedMarkersCenters = weightedEdgesExtractor.DetectMarkerCenters(image(region), cv::Size2i(20, 20));
    std::string line;
    for (const auto& center : refinedMarkersCenters)
    {
        // (0, 0) marks a coarse centre for which no refined contour was found.
        if (center.x == 0.0 && center.y == 0.0)
        {
            continue;
        }
        line += "(" + FormatCoordinate(center.x + region.x) + ";" + FormatCoordinate(center.y + region.y) + "),";
    }
    return line.empty() ? std::string("(0;0)") : line;
}

void WriteFrameLines(JsonTrackerObject& jsonObject, const std::string& sectionName, const std::vector<std::string>& lines)
{
    for (size_t i = 0; i < lines.size(); ++i)
    {
        jsonObject[sectionName][BuildFrameKey(i)] = lines[i];
    }
}
}   // namespace

void MultiBlobDetectorTracker::SetLabelledDetection(const cv::SimpleBlobDetector::Params& params, const cv::Rect& region)
{
    circleBlobDetectorParams = params;
    detectionRegion = region;
    labelled = true;
}

void MultiBlobDetectorTracker::RunTracking(JsonTrackerObject &jsonObject)
{
    const std::string sectionName = "MultiBlobDetector";
    auto blobDetector = cv::SimpleBlobDetector::create(circleBlobDetectorParams);
    cv::Rect region;
    for (size_t i = 0; i < sequence.size(); ++i)
    {
        const auto& image = sequence.at(i);
        region = SearchRegion(detectionRegion, image);
        const auto keypoints = DetectBlobs(*blobDetector, image, region);

        cv::Mat show;
        if (RunConfig().showWindows)
        {
            show = image.clone();
            for (size_t j = 0; j < keypoints.size(); ++j)
            {
                // Rounded only to address pixels for drawing; the results keep the sub-pixel centre.
                const cv::Point drawCentre(cvRound(keypoints[j].pt.x), cvRound(keypoints[j].pt.y));
                cv::circle(show, drawCentre, 3, cv::Scalar(0, 255, 0), -1, 8, 0);
                cv::circle(show, drawCentre, circleRadius, cv::Scalar(0, 0, 255), 3, 8, 0);
                cv::putText(show, cv::String(std::to_string(j)), drawCentre, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            }
        }

        const auto frameNumber = BuildFrameKey(i);
        jsonObject[sectionName][frameNumber] = BlobFrameLine(keypoints);
        if (RunConfig().logEveryFrame)
        {
            std::cout << sectionName << " - " << frameNumber << std::endl;
        }
        if (RunConfig().showWindows)
        {
            cv::imshow("Tracker result. Press 'x' to quit", show);
            if (cv::waitKey(1) == 27) break;
        }
    }
    DescribeReference(jsonObject, sectionName, labelled, markerPreset, circleBlobDetectorParams, region);
}

void MultiBlobDetectorTracker::DetectFrames(size_t begin, size_t end, std::vector<std::string>& lines) const
{
    // One detector per range: SimpleBlobDetector keeps per-call state, so instances are not shared.
    auto blobDetector = cv::SimpleBlobDetector::create(circleBlobDetectorParams);
    for (size_t i = begin; i < end; ++i)
    {
        lines[i] = BlobFrameLine(DetectBlobs(*blobDetector, sequence[i], SearchRegion(detectionRegion, sequence[i])));
    }
}

void MultiBlobDetectorTracker::WriteFrames(const std::vector<std::string>& lines, JsonTrackerObject& jsonObject) const
{
    const std::string sectionName = "MultiBlobDetector";
    WriteFrameLines(jsonObject, sectionName, lines);
    DescribeReference(jsonObject, sectionName, labelled, markerPreset, circleBlobDetectorParams,
        SearchRegion(detectionRegion, sequence.front()));
}


void MultiBlobDetectorTrackerReference::InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius,
    MarkerPreset preset)
{
    this->ROIs = boundingBoxes;
    this->sequence = images;
    this->circleRadius = radius;
    this->markerPreset = preset;
    circleBlobDetectorParams = BlobParams(preset);
}

void MultiBlobDetectorTrackerReference::SetLabelledDetection(const cv::SimpleBlobDetector::Params& params, const cv::Rect& region)
{
    circleBlobDetectorParams = params;
    detectionRegion = region;
    labelled = true;
}

void MultiBlobDetectorTrackerReference::RunTracking(JsonTrackerObject& jsonObject)
{
    const std::string sectionName = "WeightedEdgesExtractor";
    WeightedEdgesExtractor::Parameters parameters;
    parameters.blobDetectorParameters = circleBlobDetectorParams;
    cv::Rect region;
    for (size_t i = 0; i < sequence.size(); ++i)
    {
        region = SearchRegion(detectionRegion, sequence.at(i));
        const auto frameNumber = BuildFrameKey(i);
        jsonObject[sectionName][frameNumber] = PcweFrameLine(parameters, sequence.at(i), region);
        if (RunConfig().logEveryFrame)
        {
            std::cout << sectionName << " - " << frameNumber << std::endl;
        }
    }
    DescribeReference(jsonObject, sectionName, labelled, markerPreset, circleBlobDetectorParams, region);
}

void MultiBlobDetectorTrackerReference::DetectFrames(size_t begin, size_t end, std::vector<std::string>& lines) const
{
    WeightedEdgesExtractor::Parameters parameters;
    parameters.blobDetectorParameters = circleBlobDetectorParams;
    for (size_t i = begin; i < end; ++i)
    {
        lines[i] = PcweFrameLine(parameters, sequence[i], SearchRegion(detectionRegion, sequence[i]));
    }
}

void MultiBlobDetectorTrackerReference::WriteFrames(const std::vector<std::string>& lines, JsonTrackerObject& jsonObject) const
{
    const std::string sectionName = "WeightedEdgesExtractor";
    WriteFrameLines(jsonObject, sectionName, lines);
    DescribeReference(jsonObject, sectionName, labelled, markerPreset, circleBlobDetectorParams,
        SearchRegion(detectionRegion, sequence.front()));
}

namespace
{
/*Resets the random state the trackers draw from: rand() (TLD, Boosting and MIL features, Boosting
    updates) and cv::theRNG() (copied by the MIL sampler) - both to the values a new process or
    thread starts with. rand() is thread-local in TrackerBench on Linux (ThreadLocalRand.cpp) and
    in the Microsoft C runtime.*/
void ResetRandomState()
{
    std::srand(1);
    cv::theRNG() = cv::RNG();
}

cv::Mat FirstFrame(VideoFrames& frames)
{
    cv::Mat frame;
    if (!frames.Next(frame))
    {
        throw std::runtime_error("the video could not be read");
    }
    return frame;
}

template <typename T> TargetTrack TrackLegacyTarget(const cv::Rect& roi, VideoFrames& frames)
{
    TargetTrack track;
    auto tracker = T::create();
    tracker->init(FirstFrame(frames), cv::Rect2d(roi));
    cv::Rect2d box(roi);
    cv::Mat image;
    while (frames.Next(image))
    {
        const bool ok = tracker->update(image, box);
        track.frames.push_back(ToTarget(box, ok));
    }
    return track;
}

template <typename T> TargetTrack TrackModernTarget(const cv::Rect& roi, VideoFrames& frames)
{
    TargetTrack track;
    auto tracker = T::create();
    tracker->init(FirstFrame(frames), roi);
    cv::Rect box;
    cv::Mat image;
    while (frames.Next(image))
    {
        const bool ok = tracker->update(image, box);
        track.frames.push_back(ToTarget(cv::Rect2d(box), ok));
    }
    return track;
}

TargetTrack TrackHybridTarget(const cv::Rect& roi, VideoFrames& frames)
{
    TargetTrack track;
    HybridTarget hybridTarget(FirstFrame(frames), roi);
    const HybridTracker::Parameters parameters;
    cv::Mat image;
    while (frames.Next(image))
    {
        bool usedFallback = false;
        track.frames.push_back(hybridTarget.Update(image, parameters, usedFallback));
        track.switched.push_back(usedFallback ? 1 : 0);
    }
    return track;
}

struct PerTargetTracker
{
    std::string section;
    TargetTrack (*track)(const cv::Rect&, VideoFrames&);
};

/*The trackers of the benchmark by their command-line name, with the section names the
    frame-by-frame run writes.*/
const std::map<std::string, PerTargetTracker>& PerTargetTrackers()
{
    static const std::map<std::string, PerTargetTracker> trackers = {
        { "MOSSE",      { GetBareTrackerName(typeid(cv::legacy::TrackerMOSSE)),      &TrackLegacyTarget<cv::legacy::TrackerMOSSE> } },
        { "CSRT",       { GetBareTrackerName(typeid(cv::legacy::TrackerCSRT)),       &TrackLegacyTarget<cv::legacy::TrackerCSRT> } },
        { "MIL",        { GetBareTrackerName(typeid(cv::legacy::TrackerMIL)),        &TrackLegacyTarget<cv::legacy::TrackerMIL> } },
        { "TLD",        { GetBareTrackerName(typeid(cv::legacy::TrackerTLD)),        &TrackLegacyTarget<cv::legacy::TrackerTLD> } },
        { "Boosting",   { GetBareTrackerName(typeid(cv::legacy::TrackerBoosting)),   &TrackLegacyTarget<cv::legacy::TrackerBoosting> } },
        { "KCF",        { GetBareTrackerName(typeid(cv::legacy::TrackerKCF)),        &TrackLegacyTarget<cv::legacy::TrackerKCF> } },
        { "MedianFlow", { GetBareTrackerName(typeid(cv::legacy::TrackerMedianFlow)), &TrackLegacyTarget<cv::legacy::TrackerMedianFlow> } },
        { "GOTURN",     { GetBareTrackerName(typeid(cv::TrackerGOTURN)),             &TrackModernTarget<cv::TrackerGOTURN> } },
        { "DaSiamRPN",  { GetBareTrackerName(typeid(cv::TrackerDaSiamRPN)),          &TrackModernTarget<cv::TrackerDaSiamRPN> } },
        { "Nano",       { GetBareTrackerName(typeid(cv::TrackerNano)),               &TrackModernTarget<cv::TrackerNano> } },
        { "ViT",        { GetBareTrackerName(typeid(cv::TrackerVit)),                &TrackModernTarget<cv::TrackerVit> } },
        { "Hybrid",     { kHybridSection,                                            &TrackHybridTarget } },
    };
    return trackers;
}
}   // namespace

VideoFrames::VideoFrames(const std::string& path, size_t maxFrames)
    : capture(path, cv::CAP_ANY, { cv::CAP_PROP_N_THREADS, 1 }), maxFrames(maxFrames)
{
}

bool VideoFrames::Next(cv::Mat& frame)
{
    if (maxFrames > 0 && framesRead >= maxFrames)
    {
        return false;
    }
    cv::Mat decoded;    // a new buffer per frame, as when the whole video is decoded up front
    if (!capture.read(decoded))
    {
        return false;
    }
    ++framesRead;
    frame = decoded;
    return true;
}

bool IsPerTargetTracker(const std::string& name)
{
    return PerTargetTrackers().count(name) > 0;
}

TargetTrack TrackSingleTarget(const std::string& name, const cv::Rect& roi, VideoFrames& frames)
{
    const auto& tracker = PerTargetTrackers().at(name);
    ResetRandomState();
    return tracker.track(roi, frames);
}

void WriteTargetTracks(const std::string& name, const std::vector<TargetTrack>& tracks, JsonTrackerObject& jsonObject)
{
    const auto& section = PerTargetTrackers().at(name).section;
    const bool hybrid = section == kHybridSection;
    const size_t frames = tracks.empty() ? 0 : tracks.front().frames.size();

    nlohmann::json validityPerFrame = nlohmann::json::object();
    nlohmann::json switchesPerFrame = nlohmann::json::object();
    long long switchCount = 0;
    for (size_t f = 0; f < frames; ++f)
    {
        std::vector<TrackedTarget> targets;
        std::vector<bool> validity;
        std::vector<bool> switched;
        for (const auto& track : tracks)
        {
            targets.push_back(track.frames[f]);
            validity.push_back(track.frames[f].valid);
            if (hybrid)
            {
                switched.push_back(track.switched[f] != 0);
                switchCount += track.switched[f] != 0 ? 1 : 0;
            }
        }
        const auto frameNumber = BuildFrameKey(f + 1);
        jsonObject[section][frameNumber] = BuildFrameLine(targets);
        validityPerFrame[frameNumber] = validity;
        if (hybrid)
        {
            switchesPerFrame[frameNumber] = switched;
        }
    }

    auto& diagnostics = jsonObject["_diagnostics"][section];
    diagnostics["n_targets"] = static_cast<int>(tracks.size());
    diagnostics["valid"] = validityPerFrame;
    if (hybrid)
    {
        diagnostics["fallback_switches"] = switchesPerFrame;
        diagnostics["fallback_switch_count"] = switchCount;
        diagnostics["primary"] = "TrackerVit";
        diagnostics["fallback"] = "TrackerCSRT";
    }
}
