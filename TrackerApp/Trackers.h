#pragma once
#include "JsonTrackerObject.h"
#include "MarkerBoxes.h"
#include "WeightedEdgesExtractor.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/tracking/tracking.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

/*Run-wide configuration for the tracking harness. Set once by whichever front end is
	driving the run (the Qt GUI or the headless benchmark harness) and read by the tracking
	templates. Deliberately Qt-free so the benchmark harness links against OpenCV alone.*/
struct TrackerRunConfig
{
	bool showWindows = true;		//!< false suppresses cv::imshow so runs work headlessly.
	bool logEveryFrame = true;		//!< false quietens the per-frame stdout chatter.
	int coordinatePrecision = 3;	//!< Decimal places used when serialising coordinates.
};

/*Accessor for the single run configuration instance.*/
TrackerRunConfig& RunConfig();

/*Serialises a coordinate with fixed precision, so sub-pixel positions are not re-quantised
	on the way out.*/
std::string FormatCoordinate(double value);

/*One tracked target in one frame, carried in double so no precision is lost between the
	tracker API and the JSON writer.*/
struct TrackedTarget
{
	double centreX = 0.0;
	double centreY = 0.0;
	double width = 0.0;
	double height = 0.0;
	bool valid = false;
};

/*Per-marker runs (TrackerBench --jobs).

	Every marker is tracked by its own tracker instance, so a run splits into independent
	(tracker, marker) jobs over the whole sequence that can run on separate threads. Each job
	starts from the default random state of the C library and of OpenCV, which TLD, Boosting
	and MIL draw their features from: the result of a job depends only on its box and the
	frames, not on which other markers or trackers ran before it or on the thread it ran on. It
	is the result of tracking that marker alone in a fresh process. No latency is measured in
	this mode; the frame-by-frame RunTracking methods below measure it.*/
struct TargetTrack
{
	std::vector<TrackedTarget> frames;	//!< Frames 1 .. n-1; frame 0 initialises the tracker.
	std::vector<char> switched;			//!< Hybrid only: frames in which the fallback took over.
};

/*The frames of a video in order, decoded as they are needed. A per-marker job reads every frame
	once, front to back, so it does not need the whole sequence in memory: memory stays the same
	for any video length, and many jobs can run side by side. Decoding uses one thread and gives
	the same frames, bit for bit, as decoding the whole video up front.*/
class VideoFrames
{
public:
	/*\param maxFrames Stop after this many frames; 0 reads to the end.*/
	VideoFrames(const std::string& path, size_t maxFrames);

	bool IsOpen() const { return capture.isOpened(); }

	/*The next frame, in a buffer of its own (trackers may keep earlier frames); false at the end.*/
	bool Next(cv::Mat& frame);

private:
	cv::VideoCapture capture;
	size_t maxFrames;
	size_t framesRead = 0;
};

/*True for the tracker names TrackSingleTarget accepts ("CSRT", "GOTURN", "Hybrid", ...).*/
bool IsPerTargetTracker(const std::string& name);

/*Tracks one marker over the whole video with a fresh instance of the named tracker, after
	resetting the random state of the calling thread. Throws when the video cannot be read.*/
TargetTrack TrackSingleTarget(const std::string& name, const cv::Rect& roi, VideoFrames& frames);

/*Writes the per-marker tracks of one tracker, in box order, as the results section the
	frame-by-frame run writes, without the latency fields.*/
void WriteTargetTracks(const std::string& name, const std::vector<TargetTrack>& tracks, JsonTrackerObject& jsonObject);

class BaseTracker
{
public:
	BaseTracker();
	~BaseTracker();

	/*Pure virtual function dedicated to inheriting classes for performing tracking.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject) = 0;
	
protected:
	std::vector<cv::Rect> ROIs;			//Vector of cv::Rect with selected regions of interests by user.
	std::vector<cv::Mat> sequence;		//Vector with sequence of images to be analyze.
	std::string sequenceFileName;
};

/*Blob Detection reference: detects every marker in every frame, independently of the trackers,
	and writes the detections to the "MultiBlobDetector" section.*/
class MultiBlobDetectorTracker : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
		\param[in] radius int value of circle to be drawn around found marker.
		\param[in] preset Detector configuration for the marker type.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius = 20,
		MarkerPreset preset = MarkerPreset::Drawn);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	void RunTracking(JsonTrackerObject &jsonObject) override;

	/*Replaces the preset with detector settings derived from the labelled markers of this video
		and restricts detection to a region (see LabelledBlobParams and ReferenceRegion).*/
	void SetLabelledDetection(const cv::SimpleBlobDetector::Params& params, const cv::Rect& region);

	/*Detections of frames [begin, end) into lines[i], in the results-file format. Frames are
		independent, so ranges can run on separate threads (TrackerBench --jobs).*/
	void DetectFrames(size_t begin, size_t end, std::vector<std::string>& lines) const;

	/*Writes the lines of every frame as this stream's results section.*/
	void WriteFrames(const std::vector<std::string>& lines, JsonTrackerObject& jsonObject) const;

private:
	cv::SimpleBlobDetector::Params circleBlobDetectorParams;
	int circleRadius;
	MarkerPreset markerPreset = MarkerPreset::Drawn;
	cv::Rect detectionRegion;	// empty: whole frame
	bool labelled = false;
};

class MultiTrackerMIL : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiHoughCirclesTracker : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerMOSSE : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerCSRT : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class TrackerGOTURNWrapper : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class TrackerDaSiamRPNWrapper : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class TrackerNanoWrapper : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject& jsonObject);
};

class MultiTrackerTLD : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerBoosting : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerKCF : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

class MultiTrackerMedianFlow : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject &jsonObject);
};

/*Point Center Weighted Extractor (PCWE): sub-pixel marker centres in every frame, written to the
	"WeightedEdgesExtractor" section. The default preset is the glued-marker configuration.*/
class MultiBlobDetectorTrackerReference : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
		\param[in] radius int value of circle to be drawn around found marker.
		\param[in] preset Coarse-detector configuration for the marker type.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images, int radius = 20,
		MarkerPreset preset = MarkerPreset::Glued);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	void RunTracking(JsonTrackerObject& jsonObject) override;

	/*Replaces the preset with detector settings derived from the labelled markers of this video
		and restricts detection to a region (see LabelledBlobParams and ReferenceRegion).*/
	void SetLabelledDetection(const cv::SimpleBlobDetector::Params& params, const cv::Rect& region);

	/*Detections of frames [begin, end) into lines[i], in the results-file format. Frames are
		independent, so ranges can run on separate threads (TrackerBench --jobs).*/
	void DetectFrames(size_t begin, size_t end, std::vector<std::string>& lines) const;

	/*Writes the lines of every frame as this stream's results section.*/
	void WriteFrames(const std::vector<std::string>& lines, JsonTrackerObject& jsonObject) const;

private:
	cv::SimpleBlobDetector::Params circleBlobDetectorParams;
	int circleRadius;
	MarkerPreset markerPreset = MarkerPreset::Glued;
	cv::Rect detectionRegion;	// empty: whole frame
	bool labelled = false;
};

/*Wrapper for OpenCV's Vision Transformer tracker (cv::TrackerVit). Requires OpenCV 4.9.0 or
	newer.*/
class TrackerVitWrapper : public BaseTracker
{
public:
	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject& jsonObject);
};

/*Cooperative hybrid pipeline: a ViT primary state-estimator paired with a CSRT fallback
	state-corrector. Both run on every frame. While the two agree, the primary's box is taken, as
	it is the more precise; when the primary reports failure or its centre lies more than about one
	marker diameter from the fallback's, the fallback's box is taken for that frame, and the primary
	is re-initialised from it when it reports failure or after ten consecutive frames of
	disagreement.

	Every switch is recorded so the benefit can be quantified rather than asserted.*/
class HybridTracker : public BaseTracker
{
public:
	/*! \struct Parameters
		Thresholds governing when the fallback takes over.
	*/
	struct Parameters
	{
		/*! Largest accepted distance between the primary's and the fallback's centres, as a
			fraction of the shorter side of the initial box. The boxes are about three marker
			diameters wide, so 1/3 is about one marker diameter: further apart, one of the two
			trackers is no longer on the marker, and the fallback is the more robust one.*/
		double maxDisagreementBoxFraction = 1.0 / 3.0;
		/*! Consecutive frames of disagreement after which the primary is re-initialised from the
			fallback's box (at once when the primary reports failure). Re-initialising inherits the
			fallback's centring error, so brief disagreements only hand over the frame.*/
		int reinitialiseAfterFrames = 10;
	};

	/*Method used to initialize the tracker.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
		\param[in] parameters Switching thresholds. Defaults are documented in Parameters.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images);

	/*Overload taking explicit switching thresholds.
		\param[in] boundingBoxes std::vector<cv::Rect> with selected bounding boxes.
		\param[in] images std::vector<cv::Mat> with loaded sequence of images.
		\param[in] parameters Switching thresholds. Defaults are documented in Parameters.
	*/
	void InitializeTracker(std::vector<cv::Rect>& boundingBoxes, std::vector<cv::Mat>& images,
		const Parameters& parameters);

	/*Method used to run tracking of markers on given sequence and ROIs.*/
	virtual void RunTracking(JsonTrackerObject& jsonObject);

private:
	Parameters hybridParameters;
};
