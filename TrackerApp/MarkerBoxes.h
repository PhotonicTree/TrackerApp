#pragma once

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <vector>

/*!	\file MarkerBoxes.h
	Initial marker boxes: blob-based proposals, interactive labelling and box files.
	Deliberately free of Qt, so the GUI application and the headless harness share it.
*/

/*Physical marker type a blob-detector configuration is tuned for.*/
enum class MarkerPreset
{
	Drawn,		//!< Hand-drawn epidermal markers.
	Glued,		//!< 3 mm glued semi-spherical markers.
	DrawnLoose	//!< Looser drawn-marker settings; only to pre-fill labelling, never for the reference.
};

/*SimpleBlobDetector parameters for one marker type.
	\param[in] preset Marker type.
	\returns Detector parameters; values not set here are OpenCV defaults.
*/
cv::SimpleBlobDetector::Params BlobParams(MarkerPreset preset);

/*Name of a preset as written into results files ("drawn" or "glued").*/
std::string PresetName(MarkerPreset preset);

/*One initial marker box, with the marker centre and diameter it was built from.*/
struct MarkerBox
{
	cv::Rect box;			//!< Initial box handed to every tracker.
	cv::Point2d centre;		//!< Marker centre in image coordinates; the box centre when unknown.
	double diameter = 0.0;	//!< Marker diameter in pixels; 0 when unknown.
	bool adjusted = false;	//!< The operator reshaped the box by hand; it is no longer a square of scale x diameter.
};

/*Reference detector settings for one video, derived from its labelled markers. The area window
	follows the labelled marker diameters - from 0.4 x the area of the 10th-percentile diameter to
	2.5 x the area of the 90th-percentile diameter - so it adapts to the apparent marker size of each
	camera; the shape limits are fixed (circularity >= 0.5, convexity >= 0.75, inertia ratio >= 0.1);
	the threshold range is that of the fixed presets. Falls back to the drawn-marker preset when no
	box carries a diameter.*/
cv::SimpleBlobDetector::Params LabelledBlobParams(const std::vector<MarkerBox>& boxes);

/*Region searched by the reference detectors: the bounding rectangle of the labelled boxes plus a
	margin on every side, clipped to the frame.*/
cv::Rect ReferenceRegion(const std::vector<MarkerBox>& boxes, const cv::Size& frameSize, int margin = 200);

/*Default box side as a multiple of the marker diameter.*/
constexpr double kDefaultBoxScale = 3.0;

/*Square box of side scale x diameter centred on a marker, clipped to the image.
	\param[in] centre Marker centre.
	\param[in] diameter Marker diameter in pixels.
	\param[in] scale Box side as a multiple of the diameter.
	\param[in] bounds Image size used for clipping.
*/
cv::Rect BoxAround(const cv::Point2d& centre, double diameter, double scale, const cv::Size& bounds);

/*Boxes around every blob the detector finds in a frame, ordered top to bottom, then left to right.
	\param[in] frame Frame to detect markers in (normally frame 0).
	\param[in] scale Box side as a multiple of the detected blob diameter.
	\param[in] preset Detector configuration.
*/
std::vector<MarkerBox> ProposeBoxes(const cv::Mat& frame, double scale,
	MarkerPreset preset = MarkerPreset::Drawn);

/*Boxes to pre-fill the labelling window. Dense layouts get the looser drawn-marker settings, which
	box most of the markers the reference settings miss at the cost of a few false proposals that
	the operator removes; sparse layouts get the reference settings, which produce no false ones.
	\param[in] frame Frame to label.
	\param[in] scale Box side as a multiple of the marker diameter.
	\param[in] expectedCount Expected number of markers; 0 if unknown, in which case a frame on
		which the reference settings already find 10 or more markers is treated as dense.
*/
std::vector<MarkerBox> LabellingProposals(const cv::Mat& frame, double scale, int expectedCount = 0);

/*Finds the marker nearest to a point, searching a small window around it. Only blobs of roughly
	the typical marker size are accepted, so hair, eyebrows and nostrils next to a marker are ignored.
	\param[in] frame Frame to search.
	\param[in] point Point near the marker, in image coordinates.
	\param[in] typicalDiameter Typical marker diameter in pixels; sets the window and size limits.
	\param[out] centre Marker centre.
	\param[out] diameter Marker diameter in pixels.
	\returns false when no marker-like blob is found in the window.
*/
bool SnapToMarker(const cv::Mat& frame, const cv::Point2d& point, double typicalDiameter,
	cv::Point2d& centre, double& diameter);

/*Changes the box size from fromScale to toScale (multiples of the marker diameter). Boxes built
	from a diameter and not reshaped are rebuilt exactly as squares of side toScale x diameter; boxes
	the operator reshaped, and boxes without a diameter, are scaled about their own centre by
	toScale / fromScale, so their shape is kept.
	\returns Number of boxes scaled proportionally.
*/
size_t RescaleBoxes(std::vector<MarkerBox>& boxes, double fromScale, double toScale, const cv::Size& bounds);

/*Box scale (multiple of the marker diameter) recorded in a box file; kDefaultBoxScale if absent.*/
double LoadBoxScale(const std::string& path);

/*The boxes alone, in order, as handed to the trackers.*/
std::vector<cv::Rect> ToRects(const std::vector<MarkerBox>& boxes);

/*Reads boxes from a box file, or from the "_rois" section of a results file.
	\returns An empty vector if the file cannot be read.
*/
std::vector<MarkerBox> LoadBoxes(const std::string& path);

/*Writes a box file, creating its directory if needed.
	\param[in] path Output file.
	\param[in] boxes Boxes, in marker order.
	\param[in] videoPath Video the boxes belong to, recorded for reference.
	\param[in] scale Box side as a multiple of the marker diameter.
	\returns false if the file cannot be written.
*/
bool SaveBoxes(const std::string& path, const std::vector<MarkerBox>& boxes,
	const std::string& videoPath, double scale);

/*Box file used for a video by convention: <labelDirectory>/<video's folder>/<video stem>.json.*/
std::string BoxFileFor(const std::string& labelDirectory, const std::string& videoPath);

/*Reads only the first frame of a video, which is all that labelling needs.*/
bool ReadFirstFrame(const std::string& videoPath, cv::Mat& frame);

/*How an interactive labelling session ended.*/
enum class LabelOutcome
{
	Saved,		//!< The operator accepted the boxes.
	Skipped,	//!< The operator skipped this video without saving.
	QuitAll		//!< The operator asked to stop labelling altogether.
};

/*Interactive labelling of marker boxes in an OpenCV window.

	Left click adds a box snapped to the nearest marker (or re-centres the box under the cursor);
	dragging an edge or corner of a box in the magnifier reshapes it into a rectangle;
	right click removes the box under the cursor, the mouse wheel and the keys 1/2/4/8 zoom at the
	cursor. Keys: a add detector proposals, A add looser proposals, u undo, c clear, +/- box size,
	Enter or s save,
	Esc skip, Q quit all, h help.

	\param[in] frame Frame to label (normally frame 0).
	\param[in,out] boxes Initial boxes on entry; the labelled boxes on return.
	\param[in,out] scale Box side as a multiple of the marker diameter; +/- change it.
	\param[in] title Window title.
	\param[in] expectedCount Expected number of markers, shown as a hint; 0 if unknown.
*/
LabelOutcome LabelBoxes(const cv::Mat& frame, std::vector<MarkerBox>& boxes, double& scale,
	const std::string& title, int expectedCount = 0);
