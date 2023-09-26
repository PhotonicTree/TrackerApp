import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

screen_width = 1920
screen_height = 1080

# Load video file
video_file = 'C:/Users/msi/Desktop/20230223_1512_Konrad4MarkersEmotionDot01/Preview_19395995.mp4'
cap = cv2.VideoCapture(video_file)

# Load JSON data
json_file = 'Results/20230223_1512_Konrad4MarkersEmotionDot01/Preview_19395995Ref.json'
with open(json_file, 'r') as f:
    data = json.load(f)

# Define the point to track (choose coordinates manually)
# Format: (x, y)
point_to_track = (707,740)

# Initialize lists to store data for the tracked point
x_blob = []
y_blob = []
x_edges = []
y_edges = []

# Initialize previous frame point
prev_point = point_to_track

# Threshold distance
distance_threshold = 10

# Iterate through frames
frame_number = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get the data for the current frame from MultiBlobDetector
    frame_key = f"{frame_number:04d}"  # Format frame number with leading zeros
    frame_data_blob = data["MultiBlobDetector"].get(frame_key)
    
    # Get the data for the current frame from WeightedEdgesExtractor
    frame_data_edges = data["WeightedEdgesExtractor"].get(frame_key)

    if frame_data_blob and frame_data_edges:
        # Parse the data for MultiBlobDetector
        points_blob = frame_data_blob.split(',')
        points_blob.pop()
        points_blob = [tuple(map(int, point.strip('()').split(';'))) for point in points_blob]
        
        # Parse the data for WeightedEdgesExtractor
        points_edges = frame_data_edges.split(',')
        points_edges.pop()
        points_edges = [tuple(map(int, point.strip('()').split(';'))) for point in points_edges]
        
        # Find the closest point in points_blob to the previous frame point
        min_distance = float('inf')
        closest_point_blob = None
        for point_blob in points_blob:
            d = np.linalg.norm(np.array(prev_point) - np.array(point_blob))
            if d < min_distance:
                min_distance = d
                closest_point_blob = point_blob
        
        # Find the closest point in points_edges to the previous frame point
        min_distance = float('inf')
        closest_point_edges = None
        for point_edges in points_edges:
            d = np.linalg.norm(np.array(prev_point) - np.array(point_edges))
            if d < min_distance:
                min_distance = d
                closest_point_edges = point_edges
        
        # Apply distance threshold
        if min_distance <= distance_threshold:
            x_blob.append(closest_point_blob[0])
            y_blob.append(closest_point_blob[1])
            x_edges.append(closest_point_edges[0])
            y_edges.append(closest_point_edges[1])
            
            # Update the previous frame point
            prev_point = closest_point_blob
    
    frame_number += 1
    
    # Display the frame with dots
    # Resize the frame to fit the screen
    cv2.circle(frame, (prev_point[0], prev_point[1]), 5, (0, 0, 255), -1)  # Draw a red dot
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    if (frame_height > screen_height):
        new_width = int(screen_height / frame_height * frame_width)
        frame = cv2.resize(frame, (new_width, screen_height))
        
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Create and save plots for the tracked point
frame_numbers = list(range(len(x_blob)))

# Plot x coordinates
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, x_blob, label="MultiBlobDetector")
plt.plot(frame_numbers, x_edges, label="WeightedEdgesExtractor")
plt.xlabel('Frame Number')
plt.ylabel('X Coordinate')
plt.title(f'X Coordinate Comparison for Tracked Point')
plt.legend()
plt.savefig('x_coordinate_plot_tracked_point.png')
plt.close()

# Plot y coordinates
plt.figure(figsize=(10, 5))
plt.plot(frame_numbers, y_blob, label="MultiBlobDetector")
plt.plot(frame_numbers, y_edges, label="WeightedEdgesExtractor")
plt.xlabel('Frame Number')
plt.ylabel('Y Coordinate')
plt.title(f'Y Coordinate Comparison for Tracked Point')
plt.legend()
plt.savefig('y_coordinate_plot_tracked_point.png')
plt.close()

# Release video capture
cap.release()

print("Plots for the tracked point have been generated and saved.")