import json
import matplotlib.pyplot as plt
import math
from os import makedirs as make_dir

TRACKERS = ["TrackerMOSSE","TrackerBoosting", "TrackerCSRT", "TrackerKCF", "TrackerMIL", "TrackerMedianFlow"]
xy_detected = {}
x_detected = {}
y_detected = {}
# Load the JSON file
filename = "Results/KonradTest.json"
with open(filename) as json_file:
    data = json.load(json_file)
    detector = data["MultiBlobDetector"]
    tracker_base = data["TrackerCSRT"]
    for frame_number , fr in zip(tracker_base, detector):
        if(frame_number != 'FPS'):
            frame_number_converted = int(frame_number)
            detector_line_values = []
            splited_detector = detector[frame_number].split(",")
        # splited_tracker = tracker_base[frame_number].split(",")
            for point in splited_detector:
                a = len(point)
                if (len(point) > 2):
                    list_of_ints = [int(x) for x in point.strip("()").split(";")]
                    detector_line_values.append(list_of_ints)
            tracker_line_values = [int(t) for t in tracker_base[frame_number].strip("()").split(";")]
            a = b = c = 0
            dist = 9999
            tracked_point_index = 0
            for i in range(len(detector_line_values)):
                a = detector_line_values[i][0] - tracker_line_values[0]
                b = detector_line_values[i][1] - tracker_line_values[1]
                c = math.sqrt(a**2 + b**2)
                if (c < dist):
                    dist = c
                    tracked_point_index = i
            xy_detected[frame_number_converted] = detector_line_values[tracked_point_index]
            x_detected[frame_number_converted] = detector_line_values[tracked_point_index][0]
            y_detected[frame_number_converted] = detector_line_values[tracked_point_index][1]
    for tracker_name in TRACKERS:
        tracker = data[tracker_name]
        distance = {}
        xy_tracked = {}
        x_tracked = {}
        y_tracked = {}
        for frame_number , fr in zip(tracker, xy_detected):
            if(frame_number != 'FPS'):
                frame_number_converted = int(frame_number)
                #splited_tracker = tracker[frame_number].split(",")
                tracker_line_values = [int(t) for t in tracker[frame_number].strip("()").split(";")]
                if(tracker_line_values[0] != 0):
                    xy_tracked[frame_number_converted] = tracker_line_values
                    a = b = c = 0
                    a = xy_detected[frame_number_converted][0] - tracker_line_values[0]
                    b = xy_detected[frame_number_converted][1] - tracker_line_values[1]
                    c = math.sqrt(a**2 + b**2)

                    distance[frame_number_converted] = c
                    x_tracked[frame_number_converted] = tracker_line_values[0]
                    y_tracked[frame_number_converted] = tracker_line_values[1]
                else:
                    xy_tracked[frame_number_converted] = xy_tracked[frame_number_converted - 1]
                    distance[frame_number_converted] = distance[frame_number_converted - 1]
                    x_tracked[frame_number_converted] = xy_tracked[frame_number_converted - 1][0]
                    y_tracked[frame_number_converted] = xy_tracked[frame_number_converted - 1][1]

        folder = filename.split('.')[0]
        try:
            make_dir(folder)
        except FileExistsError:
            print(f"Directory {folder} already exists.")

        # Plot the data
        plt.clf()
        t = list(distance.keys())
        y = list(distance.values())
        plt.plot(t, y, label="dist")
        plt.xlabel("Frame number")
        plt.ylabel("Distance (pixels)")
        plt.legend()
        #plt.show()
        plotname = folder + '/' + tracker_name + "_dist.png" 
        plt.savefig(plotname, dpi=500)

        plt.clf()
        y1 = list(y_detected.values())
        y2 = list(y_tracked.values())
        plt.plot(t, y1, label='y_detected')
        plt.plot(t, y2, label='y_tracked')
        plt.legend(loc='upper right')
        plotname = folder + '/' + tracker_name + "_yvals.png"
        plt.savefig(plotname, dpi=500)
        #plt.show()

        plt.clf()
        x1 = list(x_detected.values())
        x2 = list(x_tracked.values())
        plt.plot(t, x1, label='x_detected')
        plt.plot(t, x2, label='x_tracked')
        plt.legend(loc='upper right')
        plotname = folder + '/' + tracker_name + "_xvals.png"
        plt.savefig(plotname, dpi=500)
        


