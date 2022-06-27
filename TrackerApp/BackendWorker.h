#pragma once

#include <QObject>
#include <opencv2/opencv.hpp>
#include <iostream>

class BackendWorker : public QObject
{
	Q_OBJECT

private:
	std::vector<cv::Mat> sequence;

signals:
	void ImportingFinished();

public:
	Q_INVOKABLE void ImportSequence(const QString& path);

	Q_INVOKABLE void DisplaySequence(const QString& path);
};

