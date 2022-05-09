#include "BackendController.h"

BackendController::BackendController(BackendWorker* backendWorker, QObject* parent) : backendWorker(backendWorker), QObject(parent)
{
	backendWorker->moveToThread(&backendThread);
	// Connect importing signal with BackendWorker's ImportSequenceAction
	connect(this, &BackendController::startImporting, backendWorker, &BackendWorker::ImportSequenceAction);
	connect(backendWorker, &BackendWorker::ImportingFinished, this, &BackendController::importingFinished);

	backendThread.start();
}

BackendController::~BackendController()
{
	backendThread.quit();
	// Wait for thread exiting
	backendThread.wait();
}
