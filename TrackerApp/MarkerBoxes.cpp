#include "MarkerBoxes.h"
#include "json.hpp"

#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>

cv::SimpleBlobDetector::Params BlobParams(MarkerPreset preset)
{
    // The following parameters are described here:
    // https://docs.opencv.org/4.x/d0/d7a/classcv_1_1SimpleBlobDetector.html
    cv::SimpleBlobDetector::Params params;

    // Filter by intensity (thresholding)
    params.thresholdStep = 1;
    params.minThreshold = 20;
    params.maxThreshold = 150;
    params.minRepeatability = 4;
    params.minDistBetweenBlobs = 10;

    // Filter by color: dark markers on lighter skin
    params.filterByColor = true;
    params.blobColor = 0;

    // Filter by area and circularity: the two filters that differ between marker types
    params.filterByArea = true;
    params.filterByCircularity = true;
    params.maxCircularity = 1.0F;
    if (preset == MarkerPreset::Drawn)
    {
        params.minArea = 100;
        params.maxArea = 1500;
        params.minCircularity = 0.80F;
    }
    else if (preset == MarkerPreset::DrawnLoose)
    {
        params.minArea = 80;
        params.maxArea = 1500;
        params.minCircularity = 0.60F;
    }
    else
    {
        params.minArea = 180;
        params.maxArea = 230;
        params.minCircularity = 0.60F;
    }

    // Filter by convexity
    params.filterByConvexity = true;
    params.minConvexity = 0.8F;

    // Filter by inertia
    // https://stackoverflow.com/questions/14770756/opencv-simpleblobdetector-filterbyinertia-meaning
    params.filterByInertia = true;
    params.minInertiaRatio = 0.25F;
    params.maxInertiaRatio = 1.01F;
    return params;
}

cv::SimpleBlobDetector::Params LabelledBlobParams(const std::vector<MarkerBox>& boxes)
{
    std::vector<double> diameters;
    for (const auto& marker : boxes)
    {
        if (marker.diameter > 0.0)
        {
            diameters.push_back(marker.diameter);
        }
    }
    auto params = BlobParams(MarkerPreset::Drawn);
    if (diameters.empty())
    {
        return params;
    }
    std::sort(diameters.begin(), diameters.end());
    const size_t last = diameters.size() - 1;
    const double small = diameters[static_cast<size_t>(0.1 * last)];
    const double large = diameters[static_cast<size_t>(std::lround(0.9 * last))];
    params.minArea = static_cast<float>(std::max(8.0, 0.4 * CV_PI * small * small / 4.0));
    params.maxArea = static_cast<float>(2.5 * CV_PI * large * large / 4.0);
    params.minCircularity = 0.5F;
    params.minConvexity = 0.75F;
    params.minInertiaRatio = 0.1F;
    return params;
}

cv::Rect ReferenceRegion(const std::vector<MarkerBox>& boxes, const cv::Size& frameSize, int margin)
{
    const cv::Rect frame(0, 0, frameSize.width, frameSize.height);
    if (boxes.empty())
    {
        return frame;
    }
    cv::Rect bounds = boxes.front().box;
    for (const auto& marker : boxes)
    {
        bounds |= marker.box;
    }
    bounds.x -= margin;
    bounds.y -= margin;
    bounds.width += 2 * margin;
    bounds.height += 2 * margin;
    return bounds & frame;
}

std::string PresetName(MarkerPreset preset)
{
    switch (preset)
    {
    case MarkerPreset::Drawn: return "drawn";
    case MarkerPreset::Glued: return "glued";
    case MarkerPreset::DrawnLoose: return "drawn-loose";
    }
    return "unknown";
}

cv::Rect BoxAround(const cv::Point2d& centre, double diameter, double scale, const cv::Size& bounds)
{
    const int side = std::max(4, static_cast<int>(std::lround(diameter * scale)));
    const cv::Rect box(static_cast<int>(std::lround(centre.x - side / 2.0)),
                       static_cast<int>(std::lround(centre.y - side / 2.0)), side, side);
    return box & cv::Rect(0, 0, bounds.width, bounds.height);
}

std::vector<MarkerBox> ProposeBoxes(const cv::Mat& frame, double scale, MarkerPreset preset)
{
    std::vector<MarkerBox> boxes;
    if (frame.empty())
    {
        return boxes;
    }
    std::vector<cv::KeyPoint> keypoints;
    cv::SimpleBlobDetector::create(BlobParams(preset))->detect(frame, keypoints);

    // Stable, reproducible ordering: OpenCV's detection order is not guaranteed.
    std::sort(keypoints.begin(), keypoints.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
        return a.pt.y != b.pt.y ? a.pt.y < b.pt.y : a.pt.x < b.pt.x; });

    for (const auto& keypoint : keypoints)
    {
        // cv::KeyPoint::size from SimpleBlobDetector is the blob diameter in pixels.
        MarkerBox marker;
        marker.centre = cv::Point2d(keypoint.pt);
        marker.diameter = keypoint.size;
        marker.box = BoxAround(marker.centre, marker.diameter, scale, frame.size());
        if (marker.box.area() > 0)
        {
            boxes.push_back(marker);
        }
    }
    return boxes;
}

namespace
{
/*Drops looser proposals that cannot be drawn markers: blobs far larger than the typical marker
    (nostrils) and blobs not surrounded by skin (shirt buttons, hair, background). Skin is taken
    as a ring around the blob that is reddish and not dark.*/
std::vector<MarkerBox> KeepFaceMarkers(const cv::Mat& frame, std::vector<MarkerBox> boxes)
{
    if (boxes.size() < 5)
    {
        return boxes;
    }
    std::vector<double> diameters;
    for (const auto& marker : boxes)
    {
        diameters.push_back(marker.diameter);
    }
    std::nth_element(diameters.begin(), diameters.begin() + diameters.size() / 2, diameters.end());
    const double typical = diameters[diameters.size() / 2];

    std::vector<MarkerBox> kept;
    for (const auto& marker : boxes)
    {
        if (marker.diameter > 1.8 * typical)
        {
            continue;
        }
        if (frame.channels() == 3)
        {
            const double r = marker.diameter;
            const cv::Rect window = cv::Rect(cvFloor(marker.centre.x - 2.5 * r), cvFloor(marker.centre.y - 2.5 * r),
                                             cvCeil(5 * r) + 1, cvCeil(5 * r) + 1)
                                  & cv::Rect(0, 0, frame.cols, frame.rows);
            cv::Mat ring(window.size(), CV_8U, cv::Scalar(0));
            const cv::Point centre(cvRound(marker.centre.x) - window.x, cvRound(marker.centre.y) - window.y);
            cv::circle(ring, centre, cvRound(2.4 * r), cv::Scalar(255), -1);
            cv::circle(ring, centre, cvRound(1.2 * r), cv::Scalar(0), -1);
            const cv::Scalar mean = cv::mean(frame(window), ring);    // B, G, R
            const bool skin = mean[2] > mean[0] + 12.0 && mean[2] >= mean[1] && mean[2] > 60.0;
            if (!skin)
            {
                continue;
            }
        }
        kept.push_back(marker);
    }
    return kept;
}
}   // namespace

std::vector<MarkerBox> LabellingProposals(const cv::Mat& frame, double scale, int expectedCount)
{
    if (expectedCount >= 10)
    {
        return KeepFaceMarkers(frame, ProposeBoxes(frame, scale, MarkerPreset::DrawnLoose));
    }
    auto boxes = ProposeBoxes(frame, scale, MarkerPreset::Drawn);
    if (expectedCount <= 0 && boxes.size() >= 10)
    {
        return KeepFaceMarkers(frame, ProposeBoxes(frame, scale, MarkerPreset::DrawnLoose));
    }
    return boxes;
}

bool SnapToMarker(const cv::Mat& frame, const cv::Point2d& point, double typicalDiameter,
    cv::Point2d& centre, double& diameter)
{
    const double typical = typicalDiameter > 0.0 ? typicalDiameter : 14.0;
    const double typicalArea = CV_PI * typical * typical / 4.0;
    const double minimumArea = std::max(8.0, 0.15 * typicalArea);
    const double maximumArea = 3.0 * typicalArea;
    const int radius = std::max(8, static_cast<int>(std::lround(1.5 * typical)));
    const cv::Rect window = cv::Rect(static_cast<int>(std::lround(point.x)) - radius,
                                     static_cast<int>(std::lround(point.y)) - radius,
                                     2 * radius + 1, 2 * radius + 1)
                          & cv::Rect(0, 0, frame.cols, frame.rows);
    if (window.area() <= 0)
    {
        return false;
    }
    cv::Mat patch;
    if (frame.channels() == 3)
    {
        cv::cvtColor(frame(window), patch, cv::COLOR_BGR2GRAY);
    }
    else
    {
        patch = frame(window).clone();
    }
    const cv::Point2d offset(window.tl());
    const cv::Point2d local = point - offset;

    /*The operator has already indicated that a marker is here, so the filters only need to
        separate it from skin texture inside a small window and can be far looser than the
        detector used for the reference.*/
    cv::SimpleBlobDetector::Params params;
    params.thresholdStep = 5;
    params.minThreshold = 10;
    params.maxThreshold = 200;
    params.minRepeatability = 2;
    params.minDistBetweenBlobs = 3;
    params.filterByColor = true;
    params.blobColor = 0;
    params.filterByArea = true;
    params.minArea = static_cast<float>(minimumArea);
    params.maxArea = static_cast<float>(maximumArea);
    params.filterByCircularity = true;
    params.minCircularity = 0.4F;
    params.filterByConvexity = true;
    params.minConvexity = 0.6F;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.1F;

    std::vector<cv::KeyPoint> keypoints;
    cv::SimpleBlobDetector::create(params)->detect(patch, keypoints);
    const cv::KeyPoint* nearest = nullptr;
    double best = std::numeric_limits<double>::max();
    for (const auto& keypoint : keypoints)
    {
        const double distance = cv::norm(cv::Point2d(keypoint.pt) - local);
        if (distance < best)
        {
            best = distance;
            nearest = &keypoint;
        }
    }
    if (nearest != nullptr)
    {
        centre = cv::Point2d(nearest->pt) + offset;
        diameter = nearest->size;
        return true;
    }

    // Fallback: the dark connected component nearest to the point after Otsu thresholding.
    cv::Mat binary, labels, stats, centroids;
    cv::threshold(patch, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    const int count = cv::connectedComponentsWithStats(binary, labels, stats, centroids, 8);
    int bestLabel = -1;
    best = std::numeric_limits<double>::max();
    for (int label = 1; label < count; ++label)
    {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < minimumArea || area > maximumArea)
        {
            continue;
        }
        const cv::Point2d componentCentre(centroids.at<double>(label, 0), centroids.at<double>(label, 1));
        const double distance = cv::norm(componentCentre - local);
        if (distance < best)
        {
            best = distance;
            bestLabel = label;
        }
    }
    if (bestLabel < 0 || best > radius)
    {
        return false;
    }
    centre = cv::Point2d(centroids.at<double>(bestLabel, 0), centroids.at<double>(bestLabel, 1)) + offset;
    diameter = 2.0 * std::sqrt(stats.at<int>(bestLabel, cv::CC_STAT_AREA) / CV_PI);
    return true;
}

size_t RescaleBoxes(std::vector<MarkerBox>& boxes, double fromScale, double toScale, const cv::Size& bounds)
{
    const double factor = fromScale > 0.0 ? toScale / fromScale : 1.0;
    size_t proportional = 0;
    for (auto& marker : boxes)
    {
        if (!marker.adjusted && marker.diameter > 0.0)
        {
            marker.box = BoxAround(marker.centre, marker.diameter, toScale, bounds);
            continue;
        }
        const cv::Point2d centre(marker.box.x + marker.box.width / 2.0, marker.box.y + marker.box.height / 2.0);
        const double width = std::max(4.0, marker.box.width * factor);
        const double height = std::max(4.0, marker.box.height * factor);
        marker.box = cv::Rect(cvRound(centre.x - width / 2.0), cvRound(centre.y - height / 2.0),
                              cvRound(width), cvRound(height))
                   & cv::Rect(0, 0, bounds.width, bounds.height);
        ++proportional;
    }
    return proportional;
}

double LoadBoxScale(const std::string& path)
{
    std::ifstream file(path);
    if (!file)
    {
        return kDefaultBoxScale;
    }
    try
    {
        nlohmann::json document;
        file >> document;
        if (document.contains("_meta") && document["_meta"].contains("box_scale")
            && document["_meta"]["box_scale"].is_number())
        {
            return document["_meta"]["box_scale"].get<double>();
        }
    }
    catch (const std::exception&)
    {
    }
    return kDefaultBoxScale;
}

std::vector<cv::Rect> ToRects(const std::vector<MarkerBox>& boxes)
{
    std::vector<cv::Rect> rects;
    rects.reserve(boxes.size());
    for (const auto& marker : boxes)
    {
        rects.push_back(marker.box);
    }
    return rects;
}

std::vector<MarkerBox> LoadBoxes(const std::string& path)
{
    std::vector<MarkerBox> boxes;
    std::ifstream file(path);
    if (!file)
    {
        return boxes;
    }
    nlohmann::json document;
    try
    {
        file >> document;
    }
    catch (const std::exception& error)
    {
        std::cerr << "Could not parse box file " << path << ": " << error.what() << "\n";
        return boxes;
    }
    const nlohmann::json& section = document.contains("_rois") ? document.at("_rois") : document;

    // Keys are numeric strings; order them numerically so marker order is preserved.
    std::map<int, MarkerBox> ordered;
    for (auto it = section.begin(); it != section.end(); ++it)
    {
        const auto& value = it.value();
        if (!value.is_object() || !value.contains("x") || !value.contains("y")
            || !value.contains("w") || !value.contains("h"))
        {
            continue;
        }
        int id = 0;
        try
        {
            id = std::stoi(it.key());
        }
        catch (const std::exception&)
        {
            continue;
        }
        MarkerBox marker;
        marker.box = cv::Rect(static_cast<int>(std::lround(value["x"].get<double>())),
                              static_cast<int>(std::lround(value["y"].get<double>())),
                              static_cast<int>(std::lround(value["w"].get<double>())),
                              static_cast<int>(std::lround(value["h"].get<double>())));
        if (value.contains("cx") && value.contains("cy"))
        {
            marker.centre = cv::Point2d(value["cx"].get<double>(), value["cy"].get<double>());
        }
        else
        {
            marker.centre = cv::Point2d(marker.box.x + marker.box.width / 2.0,
                                        marker.box.y + marker.box.height / 2.0);
        }
        marker.diameter = value.contains("d") ? value["d"].get<double>() : 0.0;
        marker.adjusted = value.contains("adjusted") && value["adjusted"].is_boolean() && value["adjusted"].get<bool>();
        ordered[id] = marker;
    }
    for (const auto& entry : ordered)
    {
        boxes.push_back(entry.second);
    }
    return boxes;
}

namespace
{
double RoundTo3(double value)
{
    return std::round(value * 1000.0) / 1000.0;
}

std::string ParentDirectory(const std::string& path)
{
    const auto slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash);
}
}   // namespace

bool SaveBoxes(const std::string& path, const std::vector<MarkerBox>& boxes,
    const std::string& videoPath, double scale)
{
    const auto directory = ParentDirectory(path);
    if (!directory.empty())
    {
        cv::utils::fs::createDirectories(directory);
    }
    nlohmann::json document;
    document["_meta"]["video"] = videoPath;
    document["_meta"]["box_scale"] = scale;
    document["_meta"]["n_boxes"] = boxes.size();
    document["_rois"] = nlohmann::json::object();
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& marker = boxes[i];
        document["_rois"][std::to_string(i)] = {
            {"x", marker.box.x}, {"y", marker.box.y}, {"w", marker.box.width}, {"h", marker.box.height},
            {"cx", RoundTo3(marker.centre.x)}, {"cy", RoundTo3(marker.centre.y)},
            {"d", RoundTo3(marker.diameter)}, {"adjusted", marker.adjusted} };
    }
    std::ofstream out(path);
    if (!out)
    {
        std::cerr << "Could not write box file: " << path << "\n";
        return false;
    }
    out << document.dump(2) << "\n";
    return static_cast<bool>(out);
}

std::string BoxFileFor(const std::string& labelDirectory, const std::string& videoPath)
{
    const auto slash = videoPath.find_last_of("/\\");
    const std::string fileName = slash == std::string::npos ? videoPath : videoPath.substr(slash + 1);
    const std::string directory = ParentDirectory(videoPath);
    const auto folderStart = directory.find_last_of("/\\");
    const std::string folder = folderStart == std::string::npos ? directory : directory.substr(folderStart + 1);
    const auto dot = fileName.find_last_of('.');
    const std::string stem = dot == std::string::npos ? fileName : fileName.substr(0, dot);

    std::string path = labelDirectory;
    if (!folder.empty())
    {
        path += "/" + folder;
    }
    return path + "/" + stem + ".json";
}

bool ReadFirstFrame(const std::string& videoPath, cv::Mat& frame)
{
    cv::VideoCapture capture(videoPath);
    if (!capture.isOpened())
    {
        return false;
    }
    return capture.read(frame) && !frame.empty();
}

// ---------------------------------------------------------------------------------------------
// Interactive labelling
// ---------------------------------------------------------------------------------------------

namespace
{
/*State of one labelling window. The window shows a canvas made of a status bar on top, the whole
    frame on the left (overview) and an enlarged region on the right (magnifier). The magnifier
    follows the cursor while it is over the overview and stays put while it is over the
    magnifier, so markers can be clicked precisely there. All mapping between canvas and image
    coordinates is a scale and an offset per panel.*/
struct LabelSession
{
    cv::Mat frame;
    std::vector<MarkerBox>* boxes = nullptr;
    double* scale = nullptr;
    std::vector<std::vector<MarkerBox>> history;
    int expected = 0;
    int barHeight = 56;
    double overviewScale = 1.0;     // canvas pixels per image pixel in the overview
    cv::Rect overview;              // overview panel on the canvas
    cv::Rect magnifier;             // magnifier panel on the canvas
    double magnifierScale = 2.0;    // canvas pixels per image pixel in the magnifier
    cv::Point2d magnifierCentre;    // image point shown at the centre of the magnifier
    std::string message;
    bool confirmMismatch = false;
    bool showHelp = false;
    bool dirty = true;
    int dragBox = -1;               // box whose edges are being dragged, or -1
    int dragEdges = 0;              // edge flags of the drag
    int hoverBox = -1;              // box whose edges are under the cursor in the magnifier
    int hoverEdges = 0;
};

enum EdgeFlag { kLeft = 1, kRight = 2, kTop = 4, kBottom = 8 };

cv::Point2d MagnifierTopLeft(const LabelSession& session)
{
    const double width = session.magnifier.width / session.magnifierScale;
    const double height = session.magnifier.height / session.magnifierScale;
    const double x = std::min(std::max(0.0, session.magnifierCentre.x - width / 2.0),
                              std::max(0.0, session.frame.cols - width));
    const double y = std::min(std::max(0.0, session.magnifierCentre.y - height / 2.0),
                              std::max(0.0, session.frame.rows - height));
    return cv::Point2d(x, y);
}

/*Image point under a canvas point; false when the canvas point is on neither panel.*/
bool ToImage(const LabelSession& session, const cv::Point& canvasPoint, cv::Point2d& imagePoint, bool& inMagnifier)
{
    inMagnifier = session.magnifier.contains(canvasPoint);
    if (inMagnifier)
    {
        imagePoint = MagnifierTopLeft(session)
                   + cv::Point2d(canvasPoint - session.magnifier.tl()) / session.magnifierScale;
        return true;
    }
    if (session.overview.contains(canvasPoint))
    {
        imagePoint = cv::Point2d(canvasPoint - session.overview.tl()) / session.overviewScale;
        return true;
    }
    return false;
}

double TypicalDiameter(const std::vector<MarkerBox>& boxes)
{
    std::vector<double> diameters;
    for (const auto& marker : boxes)
    {
        if (marker.diameter > 0.0)
        {
            diameters.push_back(marker.diameter);
        }
    }
    if (diameters.empty())
    {
        return 14.0;
    }
    std::nth_element(diameters.begin(), diameters.begin() + diameters.size() / 2, diameters.end());
    return diameters[diameters.size() / 2];
}

/*Index of the box containing the point whose centre is nearest to it, or -1.*/
int BoxAt(const std::vector<MarkerBox>& boxes, const cv::Point2d& point)
{
    int found = -1;
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const auto& box = boxes[i].box;
        if (point.x < box.x || point.y < box.y || point.x > box.x + box.width || point.y > box.y + box.height)
        {
            continue;
        }
        const double distance = cv::norm(boxes[i].centre - point);
        if (distance < best)
        {
            best = distance;
            found = static_cast<int>(i);
        }
    }
    return found;
}

void Remember(LabelSession& session)
{
    session.history.push_back(*session.boxes);
    if (session.history.size() > 500)
    {
        session.history.erase(session.history.begin());
    }
    session.confirmMismatch = false;
}

void AddOrRecentre(LabelSession& session, const cv::Point2d& point)
{
    const double typical = TypicalDiameter(*session.boxes);
    cv::Point2d centre = point;
    double diameter = typical;
    const bool snapped = SnapToMarker(session.frame, point, typical, centre, diameter);

    MarkerBox marker;
    marker.centre = centre;
    marker.diameter = diameter;
    marker.box = BoxAround(centre, diameter, *session.scale, session.frame.size());
    if (marker.box.area() <= 0)
    {
        return;
    }
    Remember(session);
    const int hit = BoxAt(*session.boxes, point);
    if (hit >= 0)
    {
        (*session.boxes)[hit] = marker;
        session.message = "Re-centred box " + std::to_string(hit) + ".";
    }
    else
    {
        session.boxes->push_back(marker);
        session.message = snapped ? "Added box " + std::to_string(session.boxes->size() - 1) + "."
                                  : "No marker found there; box placed at the click.";
    }
    session.dirty = true;
}

void RemoveAt(LabelSession& session, const cv::Point2d& point)
{
    const int hit = BoxAt(*session.boxes, point);
    if (hit < 0)
    {
        return;
    }
    Remember(session);
    session.boxes->erase(session.boxes->begin() + hit);
    session.message = "Removed box " + std::to_string(hit) + "; later boxes were renumbered.";
    session.dirty = true;
}

void AddProposals(LabelSession& session, MarkerPreset preset)
{
    const auto proposals = ProposeBoxes(session.frame, *session.scale, preset);
    Remember(session);
    int added = 0;
    for (const auto& proposal : proposals)
    {
        bool duplicate = false;
        for (const auto& existing : *session.boxes)
        {
            const double minimumSeparation = std::max(4.0, 0.75 * std::max(existing.diameter, proposal.diameter));
            if (cv::norm(existing.centre - proposal.centre) < minimumSeparation)
            {
                duplicate = true;
                break;
            }
        }
        if (!duplicate)
        {
            session.boxes->push_back(proposal);
            ++added;
        }
    }
    session.message = "Added " + std::to_string(added) + " detector proposal(s).";
    session.dirty = true;
}

/*Box whose border passes within tolerance (image pixels) of a point, and which of its edges; the
    nearest border wins. Returns -1 when no border is close enough.*/
int EdgesAt(const std::vector<MarkerBox>& boxes, const cv::Point2d& point, double tolerance, int& edges)
{
    int found = -1;
    double best = std::numeric_limits<double>::max();
    edges = 0;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const cv::Rect& box = boxes[i].box;
        const double x0 = box.x, x1 = box.x + box.width, y0 = box.y, y1 = box.y + box.height;
        if (point.x < x0 - tolerance || point.x > x1 + tolerance || point.y < y0 - tolerance || point.y > y1 + tolerance)
        {
            continue;
        }
        const double left = std::abs(point.x - x0), right = std::abs(point.x - x1);
        const double top = std::abs(point.y - y0), bottom = std::abs(point.y - y1);
        int flags = 0;
        double distance = std::numeric_limits<double>::max();
        if (std::min(left, right) <= tolerance)
        {
            flags |= left <= right ? kLeft : kRight;
            distance = std::min(left, right);
        }
        if (std::min(top, bottom) <= tolerance)
        {
            flags |= top <= bottom ? kTop : kBottom;
            distance = std::min(distance, std::min(top, bottom));
        }
        if (flags != 0 && distance < best)
        {
            best = distance;
            found = static_cast<int>(i);
            edges = flags;
        }
    }
    return found;
}

void DragTo(LabelSession& session, const cv::Point2d& point)
{
    MarkerBox& marker = (*session.boxes)[session.dragBox];
    double x0 = marker.box.x, x1 = marker.box.x + marker.box.width;
    double y0 = marker.box.y, y1 = marker.box.y + marker.box.height;
    if (session.dragEdges & kLeft) x0 = std::min(point.x, x1 - 4.0);
    if (session.dragEdges & kRight) x1 = std::max(point.x, x0 + 4.0);
    if (session.dragEdges & kTop) y0 = std::min(point.y, y1 - 4.0);
    if (session.dragEdges & kBottom) y1 = std::max(point.y, y0 + 4.0);
    const cv::Rect box = cv::Rect(cvRound(x0), cvRound(y0), cvRound(x1) - cvRound(x0), cvRound(y1) - cvRound(y0))
                       & cv::Rect(0, 0, session.frame.cols, session.frame.rows);
    if (box.width >= 4 && box.height >= 4 && box != marker.box)
    {
        marker.box = box;
        marker.adjusted = true;
        session.dirty = true;
    }
}

void EndDrag(LabelSession& session)
{
    if (session.dragBox < 0)
    {
        return;
    }
    const auto& box = (*session.boxes)[session.dragBox].box;
    session.message = "Box " + std::to_string(session.dragBox) + " is now " + std::to_string(box.width)
                    + " x " + std::to_string(box.height) + " px.";
    session.dragBox = -1;
    session.dragEdges = 0;
    session.dirty = true;
}

void SetMagnification(LabelSession& session, double scale)
{
    session.magnifierScale = std::min(16.0, std::max(1.0, scale));
    session.dirty = true;
}

void OnMouse(int event, int x, int y, int flags, void* userdata)
{
    auto& session = *static_cast<LabelSession*>(userdata);
    // A drag ends on button release, or on any move without the button (released outside the window).
    if (session.dragBox >= 0 && (event == cv::EVENT_LBUTTONUP
                                 || (event == cv::EVENT_MOUSEMOVE && !(flags & cv::EVENT_FLAG_LBUTTON))))
    {
        EndDrag(session);
        return;
    }
    cv::Point2d imagePoint;
    bool inMagnifier = false;
    if (!ToImage(session, cv::Point(x, y), imagePoint, inMagnifier))
    {
        return;
    }
    if (session.dragBox >= 0 && event == cv::EVENT_MOUSEMOVE)
    {
        DragTo(session, imagePoint);
        return;
    }
    if (event == cv::EVENT_MOUSEMOVE)
    {
        if (inMagnifier)
        {
            int edges = 0;
            const int box = EdgesAt(*session.boxes, imagePoint, 6.0 / session.magnifierScale, edges);
            if (box != session.hoverBox || edges != session.hoverEdges)
            {
                session.hoverBox = box;
                session.hoverEdges = edges;
                session.dirty = true;
            }
        }
        else
        {
            session.magnifierCentre = imagePoint;     // the magnifier follows the cursor over the overview
            session.hoverBox = -1;
            session.dirty = true;
        }
    }
    switch (event)
    {
    case cv::EVENT_LBUTTONDOWN:
    {
        int edges = 0;
        const int box = inMagnifier ? EdgesAt(*session.boxes, imagePoint, 6.0 / session.magnifierScale, edges) : -1;
        if (box >= 0)
        {
            Remember(session);
            session.dragBox = box;
            session.dragEdges = edges;
            session.message = "Resizing box " + std::to_string(box) + ".";
            session.dirty = true;
        }
        else
        {
            AddOrRecentre(session, imagePoint);
        }
        break;
    }
    case cv::EVENT_RBUTTONDOWN:
        RemoveAt(session, imagePoint);
        break;
    case cv::EVENT_MOUSEWHEEL:
        SetMagnification(session, cv::getMouseWheelDelta(flags) > 0 ? session.magnifierScale * 2.0
                                                                    : session.magnifierScale / 2.0);
        break;
    default:
        break;
    }
}

void DrawBoxes(cv::Mat& panel, const LabelSession& session, const cv::Point2d& topLeft, double k,
    int thickness, double fontScale, bool showHover)
{
    auto toPanel = [&](const cv::Point2d& p) {
        const cv::Point2d q = (p - topLeft) * k;
        return cv::Point(cvRound(q.x), cvRound(q.y));
    };
    for (size_t i = 0; i < session.boxes->size(); ++i)
    {
        const auto& marker = (*session.boxes)[i];
        // green: square of scale x diameter; azure: reshaped by the operator; yellow: no diameter
        const cv::Scalar colour = marker.adjusted ? cv::Scalar(255, 190, 40)
                                : marker.diameter > 0.0 ? cv::Scalar(0, 220, 0) : cv::Scalar(0, 220, 220);
        const cv::Point a = toPanel(cv::Point2d(marker.box.tl()));
        const cv::Point b = toPanel(cv::Point2d(marker.box.br()));
        cv::rectangle(panel, a, b, colour, thickness, cv::LINE_AA);
        const int activeBox = session.dragBox >= 0 ? session.dragBox : session.hoverBox;
        const int activeEdges = session.dragBox >= 0 ? session.dragEdges : session.hoverEdges;
        if (showHover && static_cast<int>(i) == activeBox)
        {
            const cv::Scalar highlight(0, 255, 255);
            if (activeEdges & kLeft) cv::line(panel, cv::Point(a.x, a.y), cv::Point(a.x, b.y), highlight, thickness + 2, cv::LINE_AA);
            if (activeEdges & kRight) cv::line(panel, cv::Point(b.x, a.y), cv::Point(b.x, b.y), highlight, thickness + 2, cv::LINE_AA);
            if (activeEdges & kTop) cv::line(panel, cv::Point(a.x, a.y), cv::Point(b.x, a.y), highlight, thickness + 2, cv::LINE_AA);
            if (activeEdges & kBottom) cv::line(panel, cv::Point(a.x, b.y), cv::Point(b.x, b.y), highlight, thickness + 2, cv::LINE_AA);
        }
        cv::circle(panel, toPanel(marker.centre), std::max(1, thickness), cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        cv::putText(panel, std::to_string(i), a + cv::Point(0, -3), cv::FONT_HERSHEY_SIMPLEX,
                    fontScale, colour, 1, cv::LINE_AA);
    }
}

void PutText(cv::Mat& image, const std::string& text, cv::Point origin, const cv::Scalar& colour, double size = 0.55)
{
    cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, size, colour, 1, cv::LINE_AA);
}

cv::Mat Render(const LabelSession& session)
{
    const int width = session.magnifier.x + session.magnifier.width;
    const int height = session.overview.y + session.overview.height;
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(45, 45, 45));

    // Overview: the whole frame, with the magnified region outlined.
    cv::Mat overview = canvas(session.overview);
    const double k0 = session.overviewScale;
    cv::Mat shrunk;
    cv::resize(session.frame, shrunk, cv::Size(cvRound(session.frame.cols * k0), cvRound(session.frame.rows * k0)),
               0, 0, cv::INTER_AREA);
    shrunk(cv::Rect(0, 0, std::min(shrunk.cols, overview.cols), std::min(shrunk.rows, overview.rows)))
        .copyTo(overview(cv::Rect(0, 0, std::min(shrunk.cols, overview.cols), std::min(shrunk.rows, overview.rows))));
    DrawBoxes(overview, session, cv::Point2d(0, 0), k0, 1, 0.35, false);
    const cv::Point2d magnifierTopLeft = MagnifierTopLeft(session);
    const cv::Point2d magnifierSize(session.magnifier.width / session.magnifierScale,
                                    session.magnifier.height / session.magnifierScale);
    cv::rectangle(overview, cv::Point(cvRound(magnifierTopLeft.x * k0), cvRound(magnifierTopLeft.y * k0)),
                  cv::Point(cvRound((magnifierTopLeft.x + magnifierSize.x) * k0),
                            cvRound((magnifierTopLeft.y + magnifierSize.y) * k0)),
                  cv::Scalar(255, 255, 255), 1);

    // Magnifier.
    cv::Mat magnifier = canvas(session.magnifier);
    const double km = session.magnifierScale;
    cv::warpAffine(session.frame, magnifier,
                   cv::Matx23d(km, 0, -km * magnifierTopLeft.x, 0, km, -km * magnifierTopLeft.y),
                   magnifier.size(), km >= 4.0 ? cv::INTER_NEAREST : cv::INTER_LINEAR,
                   cv::BORDER_CONSTANT, cv::Scalar(45, 45, 45));
    DrawBoxes(magnifier, session, magnifierTopLeft, km, 2, 0.6, true);

    // Status bar.
    const bool countMatches = session.expected <= 0 || static_cast<int>(session.boxes->size()) == session.expected;
    std::string status = "boxes: " + std::to_string(session.boxes->size());
    if (session.expected > 0)
    {
        status += " / expected " + std::to_string(session.expected);
    }
    char settings[128];
    std::snprintf(settings, sizeof(settings),
                  "    box: %.2f x diameter    magnifier: %.0fx    Enter: save   Esc: skip   Q: stop   h: help",
                  *session.scale, session.magnifierScale);
    status += settings;
    PutText(canvas, status, cv::Point(10, 22), countMatches ? cv::Scalar(255, 255, 255) : cv::Scalar(80, 180, 255));
    PutText(canvas, session.message, cv::Point(10, 46), cv::Scalar(170, 255, 170));

    if (session.showHelp)
    {
        const std::vector<std::string> help = {
            "left click: add a box on the marker (snaps to its centre); on a box: re-centre it",
            "drag an edge or corner of a box in the magnifier: reshape it (azure = reshaped)",
            "right click: remove the box under the cursor",
            "the magnifier follows the cursor over the left panel and stops when you move into it",
            "mouse wheel or 1 / 2 / 4 / 8: magnification",
            "a / A: add detector proposals (reference / looser settings)",
            "u: undo    c: clear all    + / -: box size",
            "Enter or s: save    Esc: skip this video    Q: stop labelling" };
        cv::Mat panel = magnifier(cv::Rect(0, 0, magnifier.cols, std::min(magnifier.rows, 20 + 26 * static_cast<int>(help.size()))));
        panel *= 0.3;
        for (size_t i = 0; i < help.size(); ++i)
        {
            PutText(magnifier, help[i], cv::Point(12, 28 + 26 * static_cast<int>(i)), cv::Scalar(255, 255, 255));
        }
    }
    return canvas;
}
}   // namespace

LabelOutcome LabelBoxes(const cv::Mat& frame, std::vector<MarkerBox>& boxes, double& scale,
    const std::string& title, int expectedCount)
{
    if (frame.empty())
    {
        return LabelOutcome::Skipped;
    }
    LabelSession session;
    session.frame = frame;
    session.boxes = &boxes;
    session.scale = &scale;
    session.expected = expectedCount;

    // Panels are laid out for a panel height of 1000 canvas pixels; the window scales the canvas.
    const int panelHeight = 1000;
    const int gap = 8;
    session.overviewScale = std::min(static_cast<double>(panelHeight) / frame.rows, 1000.0 / frame.cols);
    session.overview = cv::Rect(0, session.barHeight, cvRound(frame.cols * session.overviewScale),
                                cvRound(frame.rows * session.overviewScale));
    const int magnifierSide = std::max(session.overview.height, 700);
    session.magnifier = cv::Rect(session.overview.width + gap, session.barHeight, magnifierSide, magnifierSide);
    session.overview.height = std::max(session.overview.height, magnifierSide);
    session.magnifierCentre = cv::Point2d(frame.cols / 2.0, frame.rows / 3.0);
    session.message = "Left click on markers; use the magnifier on the right for precise clicks.";

    const cv::Size canvasSize(session.magnifier.x + session.magnifier.width,
                              session.overview.y + session.overview.height);
    cv::namedWindow(title, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    const double windowScale = std::min(1.0, std::min(1700.0 / canvasSize.width, 960.0 / canvasSize.height));
    cv::resizeWindow(title, cvRound(canvasSize.width * windowScale), cvRound(canvasSize.height * windowScale));
    cv::setMouseCallback(title, OnMouse, &session);

    /*Closing the window with its title-bar button counts as a skip where the backend can report
        it. The GTK backend of OpenCV 4.10 cannot (WND_PROP_VISIBLE is always -1), so there the
        window is simply redrawn and the keys Enter, Esc and Q are the way out.*/
    bool visibilityReported = false;
    int framesSinceRedraw = 0;
    LabelOutcome outcome = LabelOutcome::Skipped;
    for (;;)
    {
        if (session.dirty || ++framesSinceRedraw >= 30)
        {
            cv::imshow(title, Render(session));
            session.dirty = false;
            framesSinceRedraw = 0;
        }
        const int key = cv::waitKey(20);
        const double visible = cv::getWindowProperty(title, cv::WND_PROP_VISIBLE);
        if (visible >= 1.0)
        {
            visibilityReported = true;
        }
        else if (visibilityReported)
        {
            break;      // window closed after it had been shown: treat as skip
        }
        if (key < 0)
        {
            continue;
        }
        const int code = key & 0xFF;
        if (code == 13 || code == 10 || code == 's')
        {
            if (boxes.empty())
            {
                session.message = "No boxes to save.";
            }
            else if (expectedCount > 0 && static_cast<int>(boxes.size()) != expectedCount
                     && !session.confirmMismatch)
            {
                session.confirmMismatch = true;
                session.message = std::to_string(boxes.size()) + " boxes, " + std::to_string(expectedCount)
                                + " expected. Press Enter again to save anyway.";
            }
            else
            {
                outcome = LabelOutcome::Saved;
                break;
            }
        }
        else if (code == 27)
        {
            outcome = LabelOutcome::Skipped;
            break;
        }
        else if (code == 'Q')
        {
            outcome = LabelOutcome::QuitAll;
            break;
        }
        else if (code == 'u')
        {
            if (!session.history.empty())
            {
                boxes = session.history.back();
                session.history.pop_back();
                session.message = "Undone.";
            }
        }
        else if (code == 'c')
        {
            Remember(session);
            boxes.clear();
            session.message = "Cleared all boxes (u to undo).";
        }
        else if (code == 'a')
        {
            AddProposals(session, MarkerPreset::Drawn);
        }
        else if (code == 'A')
        {
            AddProposals(session, MarkerPreset::DrawnLoose);
        }
        else if (code == '+' || code == '=' || code == '-' || code == '_')
        {
            Remember(session);
            const double previous = scale;
            scale = std::max(1.0, scale + ((code == '+' || code == '=') ? 0.25 : -0.25));
            RescaleBoxes(boxes, previous, scale, frame.size());
            session.message = "Box side set to " + std::to_string(scale).substr(0, 4) + " x marker diameter.";
        }
        else if (code == '1' || code == '2' || code == '4' || code == '8')
        {
            SetMagnification(session, static_cast<double>(code - '0'));
        }
        else if (code == 'h')
        {
            session.showHelp = !session.showHelp;
        }
        session.dirty = true;
    }
    cv::destroyWindow(title);
    cv::waitKey(1);
    return outcome;
}
