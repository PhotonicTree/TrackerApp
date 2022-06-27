#include "Tracker.h"

Tracker::Tracker(MarkerPosition initPosition)
{
}

Tracker::Tracker(MarkerPosition initPosition, const std::filesystem::path& blobDetectorParamsFilePath)
{
}

MarkerTrace Tracker::GetMarkerTrace()
{
    return MarkerTrace();
}

void Tracker::Track(std::vector<cv::Mat>& inputImageSequence, size_t beginFrameIndex, size_t numberOfFramesToProcess)
{
}
