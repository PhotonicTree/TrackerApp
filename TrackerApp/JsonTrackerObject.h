#pragma once
#include "json.hpp"

class JsonTrackerObject : public nlohmann::json
{
public:
	JsonTrackerObject();
	~JsonTrackerObject();
};

