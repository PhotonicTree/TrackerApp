#pragma once
#include "BackendWorker.h"
#include <QThread>

/*!	\file BackendController.h
	This file contains BackendController class for creating all connections with QML frontend and heavy operations on another thread.
*/

class BackendController : public QObject
{
	Q_OBJECT
public:
	/*BackendController constructor.
		\param[in] backendWorker Backend worker class.
		\param[in] parent Parent QObject.
	*/
	BackendController(BackendWorker* backendWorker, QObject* parent = nullptr);

	//Destructor.
	~BackendController();

signals:
	//Signal used to emit to start importing action.
	void startImporting(const QString& path);

	//Signal used to emit to start displaying action.
	void startDisplaying();

	//Signal emitted when sequence import is finished.
	void importingFinished();

private:
	BackendWorker* backendWorker; //!< A pointer to Backend Worker, which is created in main file and do all the backend methods.
	QThread backendThread;		  //!< QThread for heavy operations, which runs simultaneously and is called from QML by connections.
};
