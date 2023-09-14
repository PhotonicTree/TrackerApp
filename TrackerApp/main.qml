import QtQuick 2.9
import QtQuick.Window 2.2
import QtQuick.Controls 2.0
import QtQuick.Layouts 1.2
import Qt.labs.platform 1.0

Window {
    id: window
    title: "Tracker"
    visible: true
    width: 1480
    height: 846
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
                    fillMode: Image.PreserveAspectFit
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
                    id: trackerBlobDetectorSwitch
                    text: "SimpleBlobDetector"
                }

                Switch {
                    id: trackerMOSSESwitch
                    text: "MOSSE"
                }
                
                Switch {
                    id: trackerCSRTSwitch
                    text: "CSRT"
                }
                
                Switch {
                    id: trackerMILSwitch
                    text: "MIL"
                }
                
                Switch {
                    id: trackerTLDSwitch
                    text: "TLD"
                }
                
                Switch {
                    id: trackerBoostingSwitch
                    text: "Boosting"
                }
                
                Switch {
                    id: trackerKCFSwitch
                    text: "KCF"
                }
                
                Switch {
                    id: trackerMedianFlowSwitch
                    text: "MedianFlow"
                }


                 Switch {
                    id: trackerGOTURNSwitch
                    text: "GOTURN"
                    //enabled: false
                }
                
                Switch {
                    id: trackerDaSiamRPNSwitch
                    text: "DaSiamRPN"
                    //enabled: false
                }

                Switch {
                    id: trackerNanoSwitch
                    text: "Nano"
                    //enabled: false
                }

                Switch {
                    id: trackerHoughCirclesSwitch
                    text: "HoughCircles"
                }

                Switch {
                    id: trackerReferenceSwitch
                    text: "Reference"
                    //enabled: false
                }
            }

            ColumnLayout {
                Layout.margins: 20
                Layout.fillHeight: true
                Button {
                    Layout.alignment: Qt.AlignBottom | Qt.AlignRight
                    text: "Run Tracking For Selected Trackers"
                    onClicked: {
                        getAllSelectedTrackers();
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
                    enabled: false
                    onClicked: {
                        backendController.startDisplaying();
                        image.visible = true
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

        Connections {
        target: backendController
        
        function onImportingFailed()
        {
           var errorPopup = Qt.createQmlObject('
           import QtQuick 2.9;
           import QtQuick.Controls 2.1;
          
           Popup { 
              id: popup; modal: true; focus: true; 
              anchors.centerIn: parent
              closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside;
              visible: false 

              Text {
                text: "Invalid video capture path! Only .mp4 and .avi files supported."
                anchors.centerIn: parent
                font.pointSize: 10
              }
          }',
          window,
          "BadVideoPopup");
           errorPopup.open();
        }
    }
    
    function getAllSelectedTrackers()
    {
        var anySelected = false;
        for (let i = 0; i < trackerSelectionColumnLayout.children.length; i++)
        {
            switchesStatus.push(trackerSelectionColumnLayout.children[i].checked);
            if (trackerSelectionColumnLayout.children[i].checked)
            {
                anySelected = true;
            }
        }
        console.log(switchesStatus);
        if (anySelected)
        {
            backendWorker.GetSelectedTrackers(switchesStatus);
        } else
        {
          var errorPopup = Qt.createQmlObject('
           import QtQuick 2.9;
           import QtQuick.Controls 2.1;
          
           Popup { 
              id: popup; modal: true; focus: true; 
              anchors.centerIn: parent
              closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside;
              visible: false 

              Text {
                text: "No selected tracking method! Select atleast one."
                anchors.centerIn: parent
                font.pointSize: 10
              }
          }',
          window,
          "BadVideoPopup");
           errorPopup.open();
        }
    }
}
