#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QQuickImageProvider>

#include "BackendController.h"
#include "BackendWorker.h"

int main(int argc, char *argv[])
{
#if defined(Q_OS_WIN)
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif

    QGuiApplication app(argc, argv);
    QQmlApplicationEngine engine;

    // set context for backendController
    BackendWorker* backendWorker(new BackendWorker);
    auto backendController = BackendController(backendWorker);
    engine.rootContext()->setContextProperty("backendController", &backendController);

    
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
