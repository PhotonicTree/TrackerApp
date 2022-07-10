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

    RowLayout {
        anchors.fill: parent
        anchors.margins: 10

        ColumnLayout {
            id: videoLoadingColumnLayout

            RowLayout {
                Layout.preferredHeight: 40
                Layout.fillWidth: true

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

        ColumnLayout {
            id: trackerSelectionColumnLayout
            Layout.margins: 200
            Layout.fillHeight: true
            Switch {
                id: trackerMOSSERadioButton
                text: "MOSSE"
            }

            Switch {
                id: trackerCSRTRadioButton
                text: "CSRT"
            }

            Button {
                Layout.alignment: Qt.AlignBottom | Qt.AlignRight
                text: "Compute"
                onClicked: {
                    backendController.startImporting(path.text);
                    //imageViewer.openVideoCamera(path.text)
                    //opencvImage.visible = true
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
                     backendController.startDisplaying();
                     image.visible = true
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
}
