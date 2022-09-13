#include "BackendController.h"

BackendController::BackendController(BackendWorker* backendWorker, QObject* parent) : backendWorker(backendWorker), QObject(parent)
{
	backendWorker->moveToThread(&backendThread);
	// Connect importing signal with BackendWorker's ImportSequenceAction
	connect(this, &BackendController::startImporting, backendWorker, &BackendWorker::ImportSequence);
	connect(backendWorker, &BackendWorker::importingFinished, this, &BackendController::importingFinished);
	connect(backendWorker, &BackendWorker::importingFailed, this, &BackendController::importingFailed);

	connect(this, &BackendController::startDisplaying, backendWorker, &BackendWorker::DisplaySequence);
	backendThread.start();
}

BackendController::~BackendController()
{
	backendThread.quit();
	// Wait for thread exiting
	backendThread.wait();
}
