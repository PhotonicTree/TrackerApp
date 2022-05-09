#pragma once

#include <opencv2/core.hpp>
#include <QObject>

//! ImageViewerLayer class forward declaration.
class ImageViewerLayer;

/*! \class ImageViewer
	Widget class for displaying cv::Mat images with OpenGL.
*/
class ImageViewer : public QObject
{
	Q_OBJECT
public:

	//! \enum ImageOrientation Image orientation.
	enum class ImageOrientation : int
	{
		Normal = 0,							//!< Normal, no rotation.
		RotateRight = 1,					//!< Rotate Right, 90 degree clockwise.
		Rotate180 = 2,						//!< Rotate 180 degree.
		RotateLeft = 3						//!< Rotate Left, 90 degree counter clockwise.
	};

	/*! Basic constructor.
		\param[in] parent QWidget parent.
	*/
	ImageViewer(QWidget* parent = nullptr);

	//! Basic destructor.
	~ImageViewer();

	// QWidget events

private:
	cv::Mat image;												//!< Image to display.
};