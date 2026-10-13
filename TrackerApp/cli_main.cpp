/*Headless benchmark harness for TrackerApp.

    The Qt front end requires a person to pick a video and drag ROI boxes, so its runs cannot be
    scripted. This entry point takes the sequence and the boxes as arguments instead, so a run is
    fully determined by its command line: no GUI, no operator in the loop, and the same tracking
    code as the application.

    Two modes:

      Labelling   TrackerBench --label --video <path> [--label-dir <dir> | --save-rois <file>]
                  Opens frame 0 in a labelling window (see LabelBoxes in MarkerBoxes.h) and saves
                  the boxes. Exit code 0 when saved, 4 when skipped, 5 when the operator asked to
                  stop labelling altogether.

      Tracking    TrackerBench --video <path> [box source] [options]
                  Runs the selected trackers from the boxes and writes one results file.

    Box sources (tracking mode, first match wins):
      --rois      <file>   Box file, or a results file whose "_rois" section is reused.
      --label-dir <dir>    Box file found by convention: <dir>/<video's folder>/<video stem>.json.
      --roi-scale <s>      Without a box file: boxes around the frame-0 Blob Detection, side = s x
                           blob diameter. With a box file: every labelled box is rebuilt with side
                           s x its marker diameter (used for the region-size sweep).

    Other options:
      --trackers  a,b,c    Comma-separated tracker names (default: all trackers, no references).
      --expect    <n>      Fail unless exactly n boxes are available.
      --targets   <a:b>    Track only boxes a..b-1 (all boxes are still recorded in "_rois").
                           Every box is tracked by its own tracker instance, so splitting the
                           boxes across processes gives the same result; it keeps trackers
                           that load one large network per box (GOTURN) within memory.
      --max-frames <n>     Truncate the sequence, for quick checks.
      --out       <file>   Output JSON path (default: derived from the video name).
      --show               Display the tracking windows (off by default).
      --reference-fixed    Reference streams with the fixed presets over the whole frame, instead
                           of settings derived from the labelled boxes within the face region.
      --threads   <n>      OpenCV thread count (default 1, the single-thread benchmark protocol;
                           -1 for the OpenCV default).
      --jobs      <n>      Per-marker jobs on n threads instead of the frame-by-frame run: every
                           (tracker, marker) pair is tracked over the whole sequence as one job,
                           and the reference streams are split into frame ranges. Each job starts
                           from the default random state, so the results are the same for any n
                           and equal to tracking each marker alone (see TargetTrack in
                           Trackers.h). Tracker jobs decode the video themselves, so memory does
                           not grow with its length; with --jobs 1, --trackers a and --targets
                           k:k+1 a process tracks one marker in a few hundred MB. OpenCV runs
                           single-threaded inside the jobs, and no latency is measured: latency
                           comes from the frame-by-frame run.
      --limit     <a=n,..> With --jobs: at most n concurrent jobs of tracker a. Defaults:
                           GOTURN=3 and DaSiamRPN=6 (about 0.86 and 0.26 GB per marker), and
                           MIL=1, because OpenCV's MIL shares scratch buffers between instances.
      --list               Print the available tracker names and exit.
*/

#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "JsonTrackerObject.h"
#include "MarkerBoxes.h"
#include "Trackers.h"

namespace
{
/*Tracker names in their default run order. Names match the keys written into the results file.*/
std::vector<std::string> DefaultTrackers()
{
    return { "MOSSE", "CSRT", "MIL", "TLD", "Boosting", "KCF", "MedianFlow",
             "GOTURN", "DaSiamRPN", "Nano", "ViT", "Hybrid" };
}

/*Reference streams: full-frame detectors that do not use the boxes for tracking.*/
std::vector<std::string> ReferenceStreams()
{
    return { "BlobDetector", "BlobDetectorGlued", "PCWE", "PCWEGlued" };
}

/*Reference detection for a video: settings derived from its labelled markers and restricted to the
    face region, or (when empty) the fixed preset of the stream over the whole frame.*/
struct ReferenceDetection
{
    bool labelled = false;
    cv::SimpleBlobDetector::Params params;
    cv::Rect region;
};

/*A reference stream configured for this video, run either frame by frame or in frame ranges.*/
struct ReferenceStream
{
    std::function<void(JsonTrackerObject&)> run;
    std::function<void(size_t, size_t, std::vector<std::string>&)> detectFrames;
    std::function<void(const std::vector<std::string>&, JsonTrackerObject&)> writeFrames;
};

template <typename T>
ReferenceStream MakeReferenceStream(std::vector<cv::Rect>& rois, std::vector<cv::Mat>& sequence,
    MarkerPreset preset, const ReferenceDetection& reference)
{
    auto stream = std::make_shared<T>();
    stream->InitializeTracker(rois, sequence, 20, preset);
    if (reference.labelled)
    {
        stream->SetLabelledDetection(reference.params, reference.region);
    }
    return { [stream](JsonTrackerObject& results) { stream->RunTracking(results); },
             [stream](size_t begin, size_t end, std::vector<std::string>& lines) { stream->DetectFrames(begin, end, lines); },
             [stream](const std::vector<std::string>& lines, JsonTrackerObject& results) { stream->WriteFrames(lines, results); } };
}

/*The reference stream of that name; false when the name is not a reference stream.*/
bool FindReferenceStream(const std::string& name, std::vector<cv::Rect>& rois, std::vector<cv::Mat>& sequence,
    const ReferenceDetection& reference, ReferenceStream& stream)
{
    if (name == "BlobDetector") stream = MakeReferenceStream<MultiBlobDetectorTracker>(rois, sequence, MarkerPreset::Drawn, reference);
    else if (name == "BlobDetectorGlued") stream = MakeReferenceStream<MultiBlobDetectorTracker>(rois, sequence, MarkerPreset::Glued, reference);
    else if (name == "PCWE") stream = MakeReferenceStream<MultiBlobDetectorTrackerReference>(rois, sequence, MarkerPreset::Drawn, reference);
    else if (name == "PCWEGlued" || name == "Reference") stream = MakeReferenceStream<MultiBlobDetectorTrackerReference>(rois, sequence, MarkerPreset::Glued, reference);
    else return false;
    return true;
}

/*Runs one named tracker against the sequence, writing into the shared results object.*/
bool RunNamedTracker(const std::string& name, std::vector<cv::Rect>& rois,
    std::vector<cv::Mat>& sequence, JsonTrackerObject& results, const ReferenceDetection& reference)
{
    ReferenceStream stream;
    if (FindReferenceStream(name, rois, sequence, reference, stream)) { stream.run(results); }
    else if (name == "MOSSE") { MultiTrackerMOSSE t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "CSRT") { MultiTrackerCSRT t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "MIL") { MultiTrackerMIL t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "TLD") { MultiTrackerTLD t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "Boosting") { MultiTrackerBoosting t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "KCF") { MultiTrackerKCF t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "MedianFlow") { MultiTrackerMedianFlow t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "GOTURN") { TrackerGOTURNWrapper t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "DaSiamRPN") { TrackerDaSiamRPNWrapper t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "Nano") { TrackerNanoWrapper t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "ViT") { TrackerVitWrapper t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else if (name == "Hybrid") { HybridTracker t; t.InitializeTracker(rois, sequence); t.RunTracking(results); }
    else { std::cerr << "Unknown tracker: " << name << "\n"; return false; }
    return true;
}

std::vector<std::string> SplitOnCommas(const std::string& text)
{
    std::vector<std::string> parts;
    std::string current;
    std::istringstream stream(text);
    while (std::getline(stream, current, ',')) { if (!current.empty()) parts.push_back(current); }
    return parts;
}

/*One unit of work of a per-marker run (--jobs).*/
struct Job
{
    std::string tracker;        //!< Tracker or stream name, for the concurrency limits.
    std::string description;    //!< For the progress output.
    double cost = 0.0;          //!< Expected duration in any unit; longest jobs start first.
    std::function<void()> run;
};

/*Cost of one marker over one frame in ms (i9-13900H, one thread). Only the order matters: the
    longest jobs start first, so no long job is left running alone at the end of a video.*/
double CostPerMarkerFrame(const std::string& name)
{
    static const std::map<std::string, double> costs = {
        { "TLD", 93.0 }, { "DaSiamRPN", 92.0 }, { "GOTURN", 58.0 }, { "Hybrid", 24.0 }, { "MIL", 20.0 },
        { "CSRT", 14.0 }, { "Boosting", 6.5 }, { "Nano", 6.5 }, { "ViT", 5.7 }, { "MedianFlow", 4.6 },
        { "KCF", 1.5 }, { "MOSSE", 0.1 } };
    const auto found = costs.find(name);
    return found == costs.end() ? 10.0 : found->second;
}

/*Runs the jobs on `workers` threads with at most limits[tracker] jobs of one tracker at a time.
    Jobs start in order of the time their tracker needs to finish: the job's own cost, or for a
    limited tracker the chain of its jobs that the limit forces to run one after another. Returns
    the first error, or an empty string when every job succeeded; after an error no further job
    is started.*/
std::string RunJobs(std::vector<Job> jobs, int workers, const std::map<std::string, int>& limits)
{
    std::map<std::string, size_t> jobsPerTracker;
    for (const auto& job : jobs) ++jobsPerTracker[job.tracker];
    for (auto& job : jobs)
    {
        const auto limit = limits.find(job.tracker);
        if (limit != limits.end())
        {
            const auto rounds = (jobsPerTracker[job.tracker] + limit->second - 1) / limit->second;
            job.cost *= static_cast<double>(rounds);
        }
    }
    std::stable_sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.cost > b.cost; });
    std::mutex mutex;
    std::condition_variable jobFinished;
    std::vector<char> started(jobs.size(), 0);
    std::map<std::string, int> running;
    size_t finished = 0;
    std::string error;

    auto worker = [&]()
    {
        std::unique_lock<std::mutex> lock(mutex);
        for (;;)
        {
            size_t next = jobs.size();
            bool waiting = false;
            for (size_t j = 0; j < jobs.size() && next == jobs.size(); ++j)
            {
                if (started[j]) continue;
                const auto limit = limits.find(jobs[j].tracker);
                if (limit == limits.end() || running[jobs[j].tracker] < limit->second) next = j;
                else waiting = true;
            }
            if (!error.empty() || (next == jobs.size() && !waiting)) return;
            if (next == jobs.size()) { jobFinished.wait(lock); continue; }

            started[next] = 1;
            ++running[jobs[next].tracker];
            lock.unlock();
            std::string failure;
            const auto jobStart = std::chrono::steady_clock::now();
            try { jobs[next].run(); }
            catch (const std::exception& exception) { failure = exception.what(); }
            catch (...) { failure = "unknown exception"; }
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - jobStart).count();
            lock.lock();

            --running[jobs[next].tracker];
            ++finished;
            if (!failure.empty() && error.empty()) error = jobs[next].description + ": " + failure;
            std::ostringstream progress;
            progress << "[" << finished << "/" << jobs.size() << "] " << jobs[next].description << "  "
                     << std::fixed << std::setprecision(1) << seconds << " s" << (failure.empty() ? "" : "  FAILED");
            std::cout << progress.str() << std::endl;
            jobFinished.notify_all();
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < std::max(1, workers); ++t) threads.emplace_back(worker);
    for (auto& thread : threads) thread.join();
    return error;
}

/*Per-marker run: every (tracker, marker) pair and every frame range of a reference stream is a
    job on a pool of `workers` threads. Tracker jobs decode the video themselves (VideoFrames);
    the reference streams use the decoded sequence, which is needed only when one is selected.
    The results file has the same sections as the frame-by-frame run, without the latency
    fields. Returns false after printing an error.*/
bool RunPerMarkerJobs(const std::vector<std::string>& selected, std::vector<cv::Rect>& rois, size_t firstTarget,
    const std::string& videoPath, size_t maxFrames, size_t expectedFrames, std::vector<cv::Mat>& sequence,
    JsonTrackerObject& results, const ReferenceDetection& reference, int workers, const std::map<std::string, int>& limits)
{
    struct StreamRun { ReferenceStream stream; std::vector<std::string> lines; };
    struct TrackerRun { std::string name; std::vector<TargetTrack> tracks; };
    std::vector<StreamRun> streamRuns;
    std::vector<TrackerRun> trackerRuns;
    std::vector<std::string> streamNames;
    for (const auto& name : selected)
    {
        ReferenceStream stream;
        if (FindReferenceStream(name, rois, sequence, reference, stream))
        {
            streamRuns.push_back({ stream, std::vector<std::string>(sequence.size()) });
            streamNames.push_back(name);
        }
        else if (IsPerTargetTracker(name)) trackerRuns.push_back({ name, std::vector<TargetTrack>(rois.size()) });
        else { std::cerr << "Unknown tracker: " << name << "\n"; return false; }
    }

    // The runs are complete before any job refers to them, so the references stay valid.
    std::vector<Job> jobs;
    const size_t framesPerJob = 20;
    for (size_t s = 0; s < streamRuns.size(); ++s)
    {
        auto& run = streamRuns[s];
        for (size_t begin = 0; begin < sequence.size(); begin += framesPerJob)
        {
            const size_t end = std::min(sequence.size(), begin + framesPerJob);
            jobs.push_back({ streamNames[s], streamNames[s] + " frames " + std::to_string(begin) + "-" + std::to_string(end - 1),
                             40.0 * static_cast<double>(end - begin),
                             [&run, begin, end]() { run.stream.detectFrames(begin, end, run.lines); } });
        }
    }
    for (auto& run : trackerRuns)
    {
        for (size_t k = 0; k < rois.size(); ++k)
        {
            jobs.push_back({ run.name, run.name + " marker " + std::to_string(firstTarget + k),
                             CostPerMarkerFrame(run.name) * static_cast<double>(expectedFrames),
                             [&run, &rois, &videoPath, maxFrames, k]()
                             {
                                 VideoFrames frames(videoPath, maxFrames);
                                 run.tracks[k] = TrackSingleTarget(run.name, rois[k], frames);
                             } });
        }
    }

    std::cout << jobs.size() << " jobs on " << workers << " threads" << std::endl;
    const auto error = RunJobs(std::move(jobs), workers, limits);
    if (!error.empty())
    {
        std::cerr << "Job failed - " << error << "\n";
        return false;
    }
    // Every job reads the same video, so every track must have the same length.
    size_t trackedFrames = 0;
    for (const auto& run : trackerRuns)
    {
        for (const auto& track : run.tracks)
        {
            if (trackedFrames != 0 && track.frames.size() != trackedFrames)
            {
                std::cerr << "Jobs read different numbers of frames from " << videoPath << ".\n";
                return false;
            }
            trackedFrames = track.frames.size();
        }
    }
    if (!trackerRuns.empty())
    {
        results["_meta"]["frames"] = static_cast<int>(trackedFrames + 1);
    }
    for (auto& run : streamRuns) run.stream.writeFrames(run.lines, results);
    for (auto& run : trackerRuns) WriteTargetTracks(run.name, run.tracks, results);
    return true;
}

/*"GOTURN=3,DaSiamRPN=6" -> {GOTURN: 3, DaSiamRPN: 6}; false on a malformed entry.*/
bool ParseLimits(const std::string& text, std::map<std::string, int>& limits)
{
    for (const auto& entry : SplitOnCommas(text))
    {
        const auto equals = entry.find('=');
        if (equals == std::string::npos) return false;
        try { limits[entry.substr(0, equals)] = std::max(1, std::stoi(entry.substr(equals + 1))); }
        catch (const std::exception&) { return false; }
    }
    return true;
}

void CreateParentDirectory(const std::string& path)
{
    const auto slash = path.find_last_of("/\\");
    if (slash != std::string::npos && slash > 0)
    {
        cv::utils::fs::createDirectories(path.substr(0, slash));
    }
}

/*Labelling mode: boxes on frame 0, saved to a box file.*/
int RunLabelling(const std::string& videoPath, const std::string& roiPath, const std::string& savePath,
    double scale, int expected)
{
    cv::Mat firstFrame;
    if (!ReadFirstFrame(videoPath, firstFrame))
    {
        std::cerr << "Could not read the first frame of: " << videoPath << "\n";
        return 1;
    }
    std::vector<MarkerBox> boxes;
    std::string source;
    if (!roiPath.empty())                         { boxes = LoadBoxes(roiPath); source = roiPath; }
    else if (cv::utils::fs::exists(savePath))     { boxes = LoadBoxes(savePath); source = savePath; }
    if (boxes.empty())
    {
        boxes = LabellingProposals(firstFrame, scale, expected);
        source = "Blob Detection proposals";
    }
    std::cout << "Labelling " << videoPath << " (" << boxes.size() << " initial boxes from "
              << source << ")\n"
              << "  In the window: left click adds a box, right click removes one, Enter saves,"
                 " Esc skips, Q stops, h shows all keys.\n" << std::flush;

    const auto outcome = LabelBoxes(firstFrame, boxes, scale, "Label markers - " + videoPath, expected);
    if (outcome == LabelOutcome::QuitAll) { std::cout << "Labelling stopped.\n"; return 5; }
    if (outcome == LabelOutcome::Skipped) { std::cout << "Skipped.\n"; return 4; }
    if (!SaveBoxes(savePath, boxes, videoPath, scale)) return 1;
    std::cout << "Saved " << boxes.size() << " boxes to " << savePath << "\n";
    return 0;
}
}   // namespace

int main(int argc, char* argv[])
{
    std::string videoPath, roiPath, labelDirectory, savePath, outputPath, trackerList;
    double roiScale = 0.0;
    size_t maxFrames = 0;
    bool showWindows = false;
    bool labelMode = false;
    bool fixedReference = false;
    int threadCount = 1;
    int jobCount = 0;
    /*GOTURN and DaSiamRPN hold a large network per marker; MIL in OpenCV 4.10 keeps scratch buffers
        in function-static variables (ClfMilBoost::update), so two MIL updates at the same time
        corrupt each other.*/
    std::map<std::string, int> jobLimits = { { "GOTURN", 3 }, { "DaSiamRPN", 6 }, { "MIL", 1 } };
    int expected = 0;
    std::string targetRange;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (arg == "--video") videoPath = next();
        else if (arg == "--rois") roiPath = next();
        else if (arg == "--label-dir") labelDirectory = next();
        else if (arg == "--save-rois") savePath = next();
        else if (arg == "--roi-scale") roiScale = std::stod(next());
        else if (arg == "--trackers") trackerList = next();
        else if (arg == "--expect") expected = std::stoi(next());
        else if (arg == "--targets") targetRange = next();
        else if (arg == "--out") outputPath = next();
        else if (arg == "--max-frames") maxFrames = static_cast<size_t>(std::stoul(next()));
        else if (arg == "--threads") threadCount = std::stoi(next());
        else if (arg == "--jobs") jobCount = std::stoi(next());
        else if (arg == "--limit")
        {
            const auto text = next();
            if (!ParseLimits(text, jobLimits)) { std::cerr << "--limit expects name=n,..., got " << text << "\n"; return 2; }
        }
        else if (arg == "--show") showWindows = true;
        else if (arg == "--label") labelMode = true;
        else if (arg == "--reference-fixed") fixedReference = true;
        else if (arg == "--list")
        {
            for (const auto& name : DefaultTrackers()) std::cout << name << "\n";
            for (const auto& name : ReferenceStreams()) std::cout << name << "   (reference stream)\n";
            return 0;
        }
        else { std::cerr << "Unknown argument: " << arg << "\n"; return 2; }
    }

    if (videoPath.empty())
    {
        std::cerr << "Usage: TrackerBench --video <path> [--label] [--rois <file> | --label-dir <dir>]"
                     " [--roi-scale <s>] [--trackers a,b,c] [--expect n] [--max-frames n]"
                     " [--out <file>] [--show] [--threads n]\n";
        return 2;
    }

    if (labelMode)
    {
        if (savePath.empty())
        {
            savePath = BoxFileFor(labelDirectory.empty() ? "Labels" : labelDirectory, videoPath);
        }
        return RunLabelling(videoPath, roiPath, savePath, roiScale > 0.0 ? roiScale : kDefaultBoxScale, expected);
    }

    if (jobCount > 0 && showWindows)
    {
        std::cerr << "--show needs the frame-by-frame run; leave out --jobs.\n";
        return 2;
    }
    /*Same rationale as the GUI entry point: the single-threaded protocol has to be set on OpenCV
        itself. Configurable so the effect of threading can itself be measured. Per-marker jobs
        already occupy the cores, so OpenCV stays single-threaded inside them.*/
    cv::setNumThreads(jobCount > 0 ? 1 : threadCount);
    RunConfig().showWindows = showWindows;
    RunConfig().logEveryFrame = false;

    /*The frame-by-frame run and the reference streams work on the decoded sequence in memory.
        Per-marker tracker jobs decode the video themselves, so for them only frame 0 is read here.*/
    const auto selected = trackerList.empty() ? DefaultTrackers() : SplitOnCommas(trackerList);
    const auto references = ReferenceStreams();
    const bool decodeAll = jobCount == 0 || std::any_of(selected.begin(), selected.end(), [&](const std::string& name)
        { return name == "Reference" || std::find(references.begin(), references.end(), name) != references.end(); });

    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened())
    {
        std::cerr << "Could not open video: " << videoPath << "\n";
        return 1;
    }
    const auto containerFrames = static_cast<size_t>(std::max(0.0, capture.get(cv::CAP_PROP_FRAME_COUNT)));
    std::vector<cv::Mat> sequence;
    cv::Mat frame;
    while (capture.read(frame))
    {
        sequence.push_back(frame.clone());
        if (!decodeAll || (maxFrames > 0 && sequence.size() >= maxFrames)) break;
    }
    capture.release();
    if (sequence.empty())
    {
        std::cerr << "No frames decoded from: " << videoPath << "\n";
        return 1;
    }
    if (decodeAll) std::cout << "Loaded " << sequence.size() << " frames from " << videoPath << "\n";
    const size_t expectedFrames = decodeAll ? sequence.size()
                                            : (maxFrames > 0 ? std::min(maxFrames, containerFrames) : containerFrames);

    std::string boxFile = roiPath;
    if (boxFile.empty() && !labelDirectory.empty())
    {
        boxFile = BoxFileFor(labelDirectory, videoPath);
        if (!cv::utils::fs::exists(boxFile))
        {
            std::cerr << "No box file for this video: " << boxFile << " (label it first with --label).\n";
            return 3;
        }
    }

    std::vector<MarkerBox> boxes;
    std::string roiSource;
    if (!boxFile.empty())
    {
        boxes = LoadBoxes(boxFile);
        roiSource = "file: " + boxFile;
        if (roiScale > 0.0)
        {
            const double labelledScale = LoadBoxScale(boxFile);
            const size_t proportional = RescaleBoxes(boxes, labelledScale, roiScale, sequence.front().size());
            roiSource += ", rescaled from " + std::to_string(labelledScale) + " to " + std::to_string(roiScale)
                       + " x marker diameter (" + std::to_string(proportional) + " reshaped box(es) scaled in proportion)";
        }
    }
    else if (roiScale > 0.0)
    {
        boxes = ProposeBoxes(sequence.front(), roiScale);
        roiSource = "blob-derived, scale=" + std::to_string(roiScale);
    }
    else
    {
        std::cerr << "No boxes: pass --rois <file>, --label-dir <dir> or --roi-scale <s>.\n";
        return 2;
    }
    if (boxes.empty())
    {
        std::cerr << "No boxes available - nothing to track.\n";
        return 1;
    }
    if (expected > 0 && static_cast<int>(boxes.size()) != expected)
    {
        std::cerr << boxes.size() << " boxes available, " << expected << " expected.\n";
        return 6;
    }
    std::cout << "Using " << boxes.size() << " box(es) from " << roiSource << "\n";

    size_t firstTarget = 0, endTarget = boxes.size();
    if (!targetRange.empty())
    {
        const auto colon = targetRange.find(':');
        try
        {
            firstTarget = std::stoul(targetRange.substr(0, colon));
            endTarget = colon == std::string::npos ? firstTarget + 1 : std::stoul(targetRange.substr(colon + 1));
        }
        catch (const std::exception&)
        {
            std::cerr << "--targets expects a:b, got " << targetRange << "\n";
            return 2;
        }
        endTarget = std::min(endTarget, boxes.size());
        if (firstTarget >= endTarget)
        {
            std::cerr << "--targets " << targetRange << " selects no box out of " << boxes.size() << ".\n";
            return 2;
        }
        std::cout << "Tracking boxes " << firstTarget << " to " << endTarget - 1 << "\n";
    }
    const std::vector<MarkerBox> trackedBoxes(boxes.begin() + firstTarget, boxes.begin() + endTarget);
    auto rois = ToRects(trackedBoxes);

    JsonTrackerObject results;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& marker = boxes[i];
        results["_rois"][std::to_string(i)] = {
            {"x", marker.box.x}, {"y", marker.box.y}, {"w", marker.box.width}, {"h", marker.box.height},
            {"cx", marker.centre.x}, {"cy", marker.centre.y}, {"d", marker.diameter},
            {"adjusted", marker.adjusted} };
    }
    results["_meta"]["opencv_version"] = CV_VERSION;
    results["_meta"]["cv_num_threads"] = cv::getNumThreads();
    results["_meta"]["cpu_features"] = cv::getCPUFeaturesLine();
    results["_meta"]["sequence"] = videoPath;
    results["_meta"]["frames"] = static_cast<int>(sequence.size());
    results["_meta"]["n_targets"] = static_cast<int>(boxes.size());
    if (!targetRange.empty())
    {
        results["_meta"]["target_range"] = { static_cast<int>(firstTarget), static_cast<int>(endTarget) };
    }
    results["_meta"]["roi_source"] = roiSource;
    results["_meta"]["coordinate_precision"] = RunConfig().coordinatePrecision;
    results["_meta"]["run_mode"] = jobCount > 0 ? "per_marker_jobs" : "frame_by_frame";
    if (jobCount > 0)
    {
        results["_meta"]["jobs"] = jobCount;
        results["_meta"]["latency"] = "not measured: per-marker jobs run concurrently";
    }

    /*With labelled boxes the reference streams use detector settings derived from those boxes and
        search only the face region; --reference-fixed restores the fixed presets over the whole frame.
        The reference uses every labelled box, also when --targets tracks a subset.*/
    ReferenceDetection reference;
    reference.labelled = !boxFile.empty() && !fixedReference;
    if (reference.labelled)
    {
        reference.params = LabelledBlobParams(boxes);
        reference.region = ReferenceRegion(boxes, sequence.front().size());
    }

    if (jobCount > 0)
    {
        if (!RunPerMarkerJobs(selected, rois, firstTarget, videoPath, maxFrames, expectedFrames, sequence, results,
                              reference, jobCount, jobLimits)) return 2;
    }
    else
    {
        for (const auto& name : selected)
        {
            std::cout << "--- " << name << " ---" << std::endl;
            if (!RunNamedTracker(name, rois, sequence, results, reference)) return 2;
        }
    }

    if (outputPath.empty())
    {
        const auto slash = videoPath.find_last_of("/\\");
        auto stem = (slash == std::string::npos) ? videoPath : videoPath.substr(slash + 1);
        const auto dot = stem.find_last_of('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        outputPath = stem + "_results.json";
    }
    CreateParentDirectory(outputPath);
    std::ofstream out(outputPath);
    if (!out)
    {
        std::cerr << "Could not write: " << outputPath << "\n";
        return 1;
    }
    out << results.dump(4);
    std::cout << "Results written to " << outputPath << std::endl;
    return 0;
}
