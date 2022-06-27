#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickImageProvider>

#include "BackendController.h"
#include "BackendWorker.h"
#include "FrameImageProvider.h"

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    // set context for backendController
    BackendWorker* backendWorker(new BackendWorker);
    FrameImageProvider* frameImageProvider(new FrameImageProvider);
    auto backendController = BackendController(backendWorker);
    engine.rootContext()->setContextProperty("backendController", &backendController);
    engine.rootContext()->setContextProperty("frameImageProvieder", frameImageProvider);

    
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
