import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.2
import Qt.labs.platform 1.0

Window {
    id: window
    title: "Tracker"
    visible: true
    width: 1260
    height: 720
    property var switchesStatus: []

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10

        ColumnLayout {
            id: videoLoadingColumnLayout

            RowLayout {
                Layout.preferredHeight: 40
                Layout.fillWidth: true
                Layout.minimumWidth: 800

                TextField {
                    id: path
                    enabled: false
                    Layout.fillWidth: true
                }

                Button {
                    text: "..."
                    onClicked: fileDialog.open();
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                border.color: "black"


                Image {
                    id: image
                    anchors {top: parent.top; fill: parent}
                    property bool counter: false
                    visible: true
                    source: "image://frameProvider/image"
                    cache: false
                    asynchronous: false 

                    function reload()
                    {
                        counter = !counter
                        source = "image://frameProvider/image?id=" + counter
                    }
                }
            }
        }

        RowLayout {

        ColumnLayout {
            id: trackerSelectionColumnLayout
            Layout.margins: 20
            Layout.fillHeight: true
            Layout.alignment: Qt.AlignRight
            Switch {
                id: trackerMOSSESwitch
                text: "MOSSE"
            }

            Switch {
                id: trackerCSRTSwitch
                text: "CSRT"
                checked: true
            }
            Switch {
                id: trackerGOTURNSwitch
                text: "GOTURN"
            }
            Switch {
                id: trackerDaSiamRPNSwitch
                text: "DaSiamRPN"
            }
            Switch {
                id: trackerHoughCirclesSwitch
                text: "HoughCircles"
            }
            Switch {
                id: tracker1
                text: "CSRT"
            }
            Switch {
                id: tracker2
                text: "CSRT"
            }
            Switch {
                id: tracker3
                text: "CSRT"
            }
            }

            ColumnLayout {
                Layout.margins: 20
                Layout.fillHeight: true
            Button {
                Layout.alignment: Qt.AlignBottom | Qt.AlignRight
                text: "Run Tracking For Selected Trackers"
                onClicked: {
                    backendController.startImporting(path.text);
                }
            }

            Button {
                Layout.alignment: Qt.AlignBottom | Qt.AlignRight
                text: "Preview"
                onClicked: {
                     imageViewer.TriggerStreamingPreview(path.text)
                     image.visible = true
                }
            }

            Button {
                Layout.alignment: Qt.AlignBottom | Qt.AlignRight
                text: "Show"
                onClicked: {
                    // backendController.startDisplaying();
                    // image.visible = true
                    getAllSelectedTrackers();
                }
            }

        }
    }
    }
    FileDialog {
        id: fileDialog
        currentFile: ""
        onFileChanged: {
            path.text = file;
        }
    }
        Connections {
        target: frameImageProvider

        function onImageChanged()
        {
            image.reload()
        }

    }

        Connections {
        target: backendController

        function onImportingFinished()
        {
           backendWorker.RunAllTrackers();
        }

    }

    function getAllSelectedTrackers()
    {
        for (let i = 0; i < trackerSelectionColumnLayout.children.length; i++)
        {
            console.log(trackerSelectionColumnLayout.children[i].checked);
            switchesStatus.push(trackerSelectionColumnLayout.children[i].checked);
            console.log(typeof(switchesStatus));
        }
        backendWorker.GetSelectedTrackers(switchesStatus);
    }
}
