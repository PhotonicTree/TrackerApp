# TrackerApp
Desktop application for easy marker tracker solutions testing.

NEURAL NETWORKS BASED TRACKERS WORK ONLY IN RELEASE MODE

To launch neural network trackers you have to download proper models and put them into your working
directory (default C:/YourRepositories/TrackerApp/TrackerApp).
Files for GOTURN tracker:
- goturn.caffemodel
- goturn.prototxt 
Both can be downloaded from opencv_extras https://github.com/opencv/opencv_extra/tree/c4219d5eb3105ed8e634278fad312a1a8d2c182d/testdata/tracking

Files for DaSiamRPN tracker:
- dasiamrpn_kernel_cls1.onnx --- https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
- dasiamrpn_kernel_r1.onnx --- https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
- dasiamrpn_model.onnx --- https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0

Files for NanoTracker:
- backbone.onnx
- neckhead.onnx
https://github.com/HonglinChu/SiamTrackers/tree/master/NanoTrack/models/nanotrackv2

All files have to have the same names (you can adjust that by setting proper tracker parameters)