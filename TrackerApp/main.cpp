/*Qt.labs.platform's FileDialog uses a native (portal/GTK) dialog where one exists and
    otherwise falls back to its Qt Widgets implementation, which requires QApplication
    rather than QGuiApplication. WSLg provides no native backend, so the fallback is the
    one that runs there. QApplication is a drop-in superset of QGuiApplication, so this
    remains correct on Windows, where the native dialog is still used.*/
#include <QApplication>
#include <opencv2/core.hpp>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickImageProvider>
#include <iostream>

#include "BackendController.h"
#include "BackendWorker.h"
#include "FrameImageProvider.h"
#include "ImageViewer.h"

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    /*Benchmarks run single-threaded to isolate algorithmic cost from hardware parallelism.
        Constraining the application to one Qt worker thread is not enough for that: OpenCV's
        internal cv::parallel_for_ and the DNN module keep their own thread pools.*/
    cv::setNumThreads(1);

    QApplication app(argc, argv);
    QQmlApplicationEngine engine;

    BackendWorker* backendWorker(new BackendWorker);
    FrameImageProvider* frameImageProvider(new FrameImageProvider);
    ImageViewer* imageViewer(new ImageViewer);
    BackendController backendController(backendWorker);

    engine.rootContext()->setContextProperty("frameImageProvider", frameImageProvider);
    engine.rootContext()->setContextProperty("imageViewer", imageViewer);
    engine.rootContext()->setContextProperty("backendWorker", backendWorker);
    engine.addImageProvider("frameProvider", frameImageProvider);
    engine.rootContext()->setContextProperty("backendController", &backendController);

    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    QObject::connect(imageViewer, &ImageViewer::newImage, frameImageProvider, &FrameImageProvider::updateImage);
    return app.exec();
}
