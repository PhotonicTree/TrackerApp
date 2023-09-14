cmake_minimum_required(VERSION 3.10)
project(TrackerApp)

# Find OpenCV package
find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Qml Gui Quick)

# Add include directories for OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add all .cpp files in the current directory to the project
file(GLOB SOURCES *.cpp)

# Create an executable
qt5_add_resources(QT_RESOURCES qml.qrc)
add_executable(TrackerApp ${SOURCES} ${QT_RESOURCES})

# Link against OpenCV
target_link_libraries(TrackerApp ${OpenCV_LIBS})
target_link_libraries(TrackerApp Qt5::Qml Qt5::Gui Qt5::Quick)