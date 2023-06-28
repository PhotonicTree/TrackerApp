#pragma once

#include <opencv2/opencv.hpp>


/*! \file WeightedEdgesExtractor.h
  Contains WeightedEdgesExtractor that is an implementation of MarkerCenterExtractor interface.
*/

/*! \class WeightedEdgesExtractor
	Class used to compute centers of markers on the image of a calibration artifact.
*/
class WeightedEdgesExtractor
{
public:

	/*! \struct Parameters
		Set of parameters for See WeightedEdgesExtractor.
	*/
	struct Parameters
	{
		/*! Default empty constructor.
		  Initializes all the parameters to default values.
		*/
		Parameters();

		int dilationSize;						/*!< Size of dilation window used to unify markers detected by SimpleBlobDetector from OpenCV library and binary image from 
													 WeightedEdgesExtractor::DifferentalBinarization. */
		int blurSize;							//!< Size of Gaussian blur applied to the image prior to detection. 
		
		int differentialBinarizationSize;		/*!< Size of window applied to DifferentalBinarization. If the value of this param is set to -1 it is computed automatically 
													 as 1 + blurSize. */
		int differentialBinarizationThreshold;	//!< Threshold value used in DifferentalBinarization. If the value of this param is set to -1 it is computed automatically. 
		
		cv::SimpleBlobDetector::Params blobDetectorParameters;	//!< Standard OpenCV SimpleBlobDetectorParams. 
	};

	/*! A simple empty constructor.
	  Currently does nothing.
	*/
	WeightedEdgesExtractor();

	/*! A constructor with parameters of an algorithm.
	  \param[in] parameters Set of parameters (WeightedEdgesExtractor::Parameters).
	*/
	explicit WeightedEdgesExtractor(
		const Parameters& parameters);

	std::vector<cv::Point2i> DetectMarkerCenters(
		const cv::Mat& image, 
		const cv::Size2i& patternSize);

protected:
	/*! Function used to refine center of each marker.
	\param[in] image Image of a single marker.
	\param[in] edges Image of binarized marker edge. See WeightedEdgesExtractor::DifferentalBinarization.
	\param[in] offset Offset between marker center in local image coordinates (single marker image) and global related to the whole image.
	\param[in] coarseCenter Center of a marker in global image coordinates computed by WeightedEdgesExtractor::DetectMarkersCoarse.
	\returns Wrapped variables of refined center of a marker in global coordinates, total calculated weight and total center replacement.
*/
	cv::Point2i RefineCenterByWeights(
		const cv::Mat& image,
		const cv::Mat& edges,
		const cv::Point2d& offset,
		const cv::Point2d& coarseCenter);

private:

	/*! Function calculates automatically threshold for WeightedEdgesExtractor::DifferentalBinarization.
	  \param[in] image Image of a pattern.
	  \param[in] differentialBinarizationSize Size of window applied to WeightedEdgesExtractor::DifferentalBinarization.
	  \param[in] blurSize Size of window of applied Gaussian blur.
	  \returns Calculated threshold.
	*/
	int CalculateThresholdForEdgeExtraction(
		const cv::Mat& image,
		int differentialBinarizationSize,
		int blurSize);

	/*! Function performs differential binarization.
	  \param[in] image Image of a pattern.
	  \param[out] binarized Binarized pattern.
	  \param[in] kernelSize Size of window.
	  \param[in] threshold Binarization threshold.
	  \returns True if completed successfully, false otherwise.
	*/
	bool DoDifferentalBinarization(
		const cv::Mat& image,
		cv::Mat& binarized,
		int kernelSize,
		int threshold);

	/*! Wrapper for detectCirclesGrid from OpenCV library.
	  \param[in] image Image of a pattern.
	  \param[in] patternSize Number of markers in each row and column of calibration artifact.
	  \param[out] centers Vector of extracted centers of markers.
	  \param[out] mask Computed mask with detected markers.
	  \returns True if completed successfully, false otherwise.
	*/
	bool DetectMarkersCoarse(
		const cv::Mat& image,
		const cv::Size2i& patternSize, 
		std::vector<cv::Point2d>& centers,
		cv::Mat& mask);

	Parameters parameters; //!< A set of parameters of the algorithm. 
};