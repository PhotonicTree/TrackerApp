cmake_minimum_required(VERSION 3.16)
project(TrackerApp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Q_OBJECT classes (BackendController, BackendWorker, FrameImageProvider, ImageViewer)
# need moc; qml.qrc needs rcc.
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# OpenCV 4.9+ is required: cv::TrackerVit does not exist before it. Qt5::Widgets is required
# because Qt.labs.platform's FileDialog falls back to its widgets implementation where no
# native dialog backend exists.
find_package(OpenCV REQUIRED)
if(OpenCV_VERSION VERSION_LESS "4.9.0")
    message(FATAL_ERROR "OpenCV ${OpenCV_VERSION} found, but >= 4.9.0 is required for cv::TrackerVit.")
endif()
find_package(Qt5 REQUIRED COMPONENTS Qml Gui Quick Widgets)

include_directories(${OpenCV_INCLUDE_DIRS})

# Tracking core: deliberately free of Qt so the headless harness links against OpenCV alone.
set(CORE_SOURCES
    Trackers.cpp
    WeightedEdgesExtractor.cpp
    JsonTrackerObject.cpp
    MarkerBoxes.cpp)

set(GUI_SOURCES
    BackendController.cpp
    BackendWorker.cpp
    FrameImageProvider.cpp
    ImageViewer.cpp)

# The interactive application.
add_executable(TrackerApp main.cpp ${CORE_SOURCES} ${GUI_SOURCES} qml.qrc)
target_link_libraries(TrackerApp ${OpenCV_LIBS})
target_link_libraries(TrackerApp Qt5::Qml Qt5::Gui Qt5::Quick Qt5::Widgets)

# The headless benchmark harness: same tracking code, no GUI and no operator in the loop,
# so benchmark runs can be scripted.
add_executable(TrackerBench cli_main.cpp ThreadLocalRand.cpp ${CORE_SOURCES})
find_package(Threads REQUIRED)
target_link_libraries(TrackerBench ${OpenCV_LIBS} Threads::Threads)
