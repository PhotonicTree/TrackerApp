#include "WeightedEdgesExtractor.h"
#include <iostream>

WeightedEdgesExtractor::Parameters::Parameters() :
	dilationSize(5), 
	blurSize(5), 
	differentialBinarizationSize(-1), 
	differentialBinarizationThreshold(-1)
{

}

WeightedEdgesExtractor::WeightedEdgesExtractor()
{

}

WeightedEdgesExtractor::WeightedEdgesExtractor(
	const Parameters& parameters)
{
	this->parameters = parameters;
}

std::vector<cv::Point2i> WeightedEdgesExtractor::DetectMarkerCenters(
	const cv::Mat& image, 
	const cv::Size2i& patternSize)
{
	if (image.empty())
	{
		return std::vector<cv::Point2i>();
	}

	if (patternSize.width <= 0 || patternSize.height <= 0)
	{
		return std::vector<cv::Point2i>();
	}

	// Convert to grayscale if necessary
	cv::Mat grayscale;
	if (image.channels() != 1)
	{
		cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
	}
	else
	{
		grayscale = image.clone();
	}

	// Optional gaussian blurr
	if (parameters.blurSize > 0)
	{
		cv::GaussianBlur(grayscale, grayscale, cv::Size(parameters.blurSize * 2 + 1, parameters.blurSize * 2 + 1), 0, 0);
	}

	std::vector<cv::Point2d> coarseCenters;
	cv::Mat mask;
	if (!DetectMarkersCoarse(grayscale, patternSize, coarseCenters, mask))
	{
		return std::vector<cv::Point2i>();
	}
	
	int differentialBinarizationSize = parameters.differentialBinarizationSize;
	int differentialBinarizationThreshold = parameters.differentialBinarizationThreshold;

	if (parameters.differentialBinarizationSize < 0)
	{
		differentialBinarizationSize = 1 + parameters.blurSize;
	}
	if (parameters.differentialBinarizationThreshold < 0)
	{
		differentialBinarizationThreshold = CalculateThresholdForEdgeExtraction(image, differentialBinarizationSize, parameters.blurSize);
	}
	
	cv::Mat binarizedImage;
	if (!DoDifferentalBinarization(grayscale, binarizedImage, differentialBinarizationSize, differentialBinarizationThreshold))
	{
		return std::vector<cv::Point2i>();
	}
	cv::bitwise_and(mask, binarizedImage, binarizedImage);

	std::vector<cv::Point2i> refinedCenters(coarseCenters.size());
	std::vector<std::vector<cv::Point>> contours;
	
	cv::findContours(binarizedImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours)
	{
		cv::Point2d localCenter;
		const int offset = 1;
		auto boundingRectangle = cv::boundingRect(contour);
		
		// Add offset
		boundingRectangle.x -= offset;
		boundingRectangle.y -= offset;
		boundingRectangle.width += 2 * offset;
		boundingRectangle.height += 2 * offset;

		size_t matchingCenterIndex = 0;
		bool matchingCenterIndexFound = false;
		for (size_t i = 0; i < coarseCenters.size(); ++i)
		{
			if (coarseCenters[i].inside(boundingRectangle))
			{
				localCenter = coarseCenters[i];
				matchingCenterIndex = i;
				matchingCenterIndexFound = true;
				break;
			}
		}

		if (matchingCenterIndexFound)
		{
			cv::Mat marker = grayscale(boundingRectangle);
			cv::Mat markerEdge = binarizedImage(boundingRectangle);
			try
			{
				refinedCenters[matchingCenterIndex] = RefineCenterByWeights(marker, markerEdge, boundingRectangle.tl(), localCenter);
			}
			catch (const std::exception& exception)
			{
				std::cout<<("Exception occurred: {}", exception.what());
			}
			
		}
	}

	return refinedCenters;
}

int WeightedEdgesExtractor::CalculateThresholdForEdgeExtraction(
	const cv::Mat& image, 
	int differentialBinarizationSize, 
	int blurSize)
{
	int differentialBinarizationThreshold = -1;

	const int histogramSize = 256;
	int channelsCount = 0;
	const float range[] = { 0.0F, 255.0F };
	const float* ranges[] = { range };
	cv::Mat histogram;

	cv::calcHist(&image, 1, &channelsCount, cv::Mat(), histogram, 1, &histogramSize, ranges);

	const int minIntensity = 0;
	const int maxIntensity = 255;
	const int intensityCorrection = 50;		// Typical black value on calibration model - expert's knowledge - p.siekanski
	const int slopeForMarkerGradient = 5;	// Typical slope for marker gradient = 5 pixels - expert's knowledge - p.siekanski
	const int pixelsInMaskMultiplier = 3;
	int whiteThreshold = image.size().area() / 5;
	double intensitySamplesCount = 0.0;
	double intensitySum = 0.0;

	for (int i = maxIntensity; i >= minIntensity; --i)
	{
		if (whiteThreshold > intensitySamplesCount)
		{
			intensitySum += static_cast<double>(histogram.at<float>(i)) * i;
			intensitySamplesCount += static_cast<double>(histogram.at<float>(i));
		}
	}
	int averageIntensity = static_cast<int>((intensitySum / intensitySamplesCount) + 0.5);
	averageIntensity -= intensityCorrection;

	int pixelGradient = static_cast<int>(averageIntensity / (static_cast<double>(blurSize) + slopeForMarkerGradient) + 0.5);
	int sumOfGradients = 0;

	for (int i = 1; i <= differentialBinarizationSize; ++i)
	{
		sumOfGradients += 2 * (differentialBinarizationSize * 2 + 1) * pixelGradient * i;
	}

	int pixelsInMask = (differentialBinarizationSize * 2 + 1) * (differentialBinarizationSize * 2 + 1);
	differentialBinarizationThreshold = static_cast<int>(static_cast<double>(sumOfGradients) / (pixelsInMaskMultiplier * static_cast<double>(pixelsInMask)) + 0.5);

	return differentialBinarizationThreshold;
}

bool WeightedEdgesExtractor::DoDifferentalBinarization(
	const cv::Mat& image, 
	cv::Mat& binarized, 
	int kernelSize, 
	int threshold)
{

	if (image.type() != CV_8UC1)
	{
		return false;
	}

	binarized = cv::Mat(image.size(), image.type());
	
	int imageHeight = image.rows;
	int imageWidth = image.cols;
	int kernelSquareSize = kernelSize * kernelSize;
	auto intensityThreshold = static_cast<double>(threshold);

	int intensityCount;
	int distance;
	double intensityDifference;
	double originalIntensity;

	for (int i = 0; i < imageHeight; ++i)
	{
		for (int j = 0; j < imageWidth; ++j)
		{
			intensityCount = 0;
			intensityDifference = 0.0;
			originalIntensity = static_cast<double>(image.ptr(i)[j]);

			for (int ii = -kernelSize; ii < kernelSize; ++ii)
			{
				for (int jj = -kernelSize; jj < kernelSize; ++jj)
				{
					distance = static_cast<int>(static_cast<uint64_t>(0.5) + static_cast<uint64_t>(ii) * static_cast<uint64_t>(ii) + static_cast<uint64_t>(jj) * static_cast<uint64_t>(jj));

					if (distance <= kernelSquareSize)
					{
						if (((i + ii) >= 0) && ((i + ii) < imageHeight) &&
							((j + jj) >= 0) && ((j + jj) < imageWidth))
						{
							intensityDifference += abs(originalIntensity - image.ptr(i + ii)[j + jj]);
							++intensityCount;
						};
					}
				}
			}

			if (intensityDifference / intensityCount >= intensityThreshold)
			{
				binarized.ptr(i)[j] = 255;
			}
			else
			{
				binarized.ptr(i)[j] = 0;
			}
		}
	}
	return true;
}

bool WeightedEdgesExtractor::DetectMarkersCoarse(
	const cv::Mat& image, 
	const cv::Size2i& patternSize, 
	std::vector<cv::Point2d>& centers, 
	cv::Mat& mask)
{
	cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(parameters.blobDetectorParameters);
	std::vector<cv::KeyPoint> keypoints;
	blobDetector->detect(image, keypoints);

	cv::Mat thresholded;
	cv::threshold(image, thresholded, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
	mask = thresholded.clone();
	int fillValue = 1;

	for (size_t i = 0; i < keypoints.size(); i++)
	{
		auto center = keypoints.at(i).pt;
		centers.push_back(center);
		cv::floodFill(mask, center, fillValue);
		++fillValue;
	}

	cv::inRange(mask, 1, fillValue, mask);
	cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(parameters.dilationSize, parameters.dilationSize));
	cv::dilate(mask, mask, structuringElement, cv::Point(-1, -1), 3);
	
	return true;
}

cv::Point2i WeightedEdgesExtractor::RefineCenterByWeights(
	const cv::Mat& grayscaleImage,
	const cv::Mat& binaryImage,
	const cv::Point2d& offset,
	const cv::Point2d& coarseCenter)
{
	const int edgeValue = 0;
	double maximumIntensity, minimumIntensity;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(grayscaleImage, &minimumIntensity, &maximumIntensity, &minLoc, &maxLoc);
	cv::Mat weightsImage(binaryImage.rows, binaryImage.cols, CV_32F);
	weightsImage.setTo(0.0f);

	double totalWeight = 0;
	double weightedSumX = 0;
	double weightedSumY = 0;

	for (int x = 0; x < binaryImage.cols - 1; ++x)
	{
		for (int y = 0; y < binaryImage.rows - 1; ++y)
		{
			if (!binaryImage.at<uchar>(y, x) == edgeValue)
			{
				auto currentIntensity = grayscaleImage.at<uchar>(y, x);
				auto pixelWeight = (1.0 - (currentIntensity - minimumIntensity) / (maximumIntensity - minimumIntensity));
				weightsImage.at<float>(y, x) = static_cast<float>(pixelWeight);
				weightedSumX += x * pixelWeight;
				weightedSumY += y * pixelWeight;
				totalWeight += pixelWeight;
			}
		}
	}
	cv::Point2i refinedCenter(0, 0);
	auto aa = (weightedSumX / totalWeight);
	refinedCenter.x = std::round(weightedSumX / totalWeight);
	refinedCenter.y = std::round(weightedSumY / totalWeight);
	refinedCenter.x += std::round(offset.x);
	refinedCenter.y += std::round(offset.y);
	return refinedCenter;
}