#include <QGuiApplication>
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

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    BackendWorker* backendWorker(new BackendWorker);
    FrameImageProvider* frameImageProvider(new FrameImageProvider);
    auto backendController = BackendController(backendWorker);
    ImageViewer* imageViewer(new ImageViewer);
    engine.rootContext()->setContextProperty("backendController", &backendController);
    engine.rootContext()->setContextProperty("frameImageProvider", frameImageProvider);
    engine.rootContext()->setContextProperty("imageViewer", imageViewer);
    engine.rootContext()->setContextProperty("backendWorker", backendWorker);
    engine.addImageProvider("frameProvider", frameImageProvider);
    
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    QObject::connect(imageViewer, &ImageViewer::newImage, frameImageProvider, &FrameImageProvider::updateImage);
    return app.exec();
}
