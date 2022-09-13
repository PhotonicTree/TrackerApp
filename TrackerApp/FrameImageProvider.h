#pragma once
#include <QObject>
#include <QImage>
#include <QQuickImageProvider>

#include <opencv2/opencv.hpp>

class FrameImageProvider : public QObject, public QQuickImageProvider
{
    Q_OBJECT
public:

    //! FrameImageProvider constructor.
    FrameImageProvider(QObject* parent = nullptr);

    //! Reimplements QQuickImageProvider::requestImage(const QString& id, QSize* size, const QSize& requestedSize)
    QImage requestImage(const QString& id, QSize* size, const QSize& requestedSize) override;

    /*! Method used to set frame index which is currently display.
        \param[in] frameIndex Current frame index to set.
    */
    void SetCurrentFrameIndex(int frameIndex);

    /*! Method used to get frame index which is currently display.
        \returns Current frame index.
    */
    int GetCurrentFrameIndex() const;


public slots:
    //! Method used to update image which is currently display.
    void updateImage(const QImage& image);

    /*! Method used to set frame to display.
         \param[in] frameIndex Frame index to display.
    */
    //void SetFrameImage(int frameIndex);

signals:
    //! Signal emitted when image to display has changed.
    void imageChanged();

private:
    QImage imageQML;                //!< QImage which is displaying.
    int currentFrameIndex;          //!< Current frame index.
};

