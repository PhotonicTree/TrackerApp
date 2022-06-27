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
                    id: firstPreviewViewerImage
                    anchors {top: parent.top; fill: parent}
                    visible: true
                    source: "image://frameImageProvieder/image"
                    cache: false
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
}
