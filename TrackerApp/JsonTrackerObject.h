#pragma once
#include "json.hpp"

class JsonTrackerObject
{
public:
	JsonTrackerObject();
	~JsonTrackerObject();
private:
	nlohmann::json jsonObject;
};

