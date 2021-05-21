# FishSense

Draft for RGBD alignment and simple length calculation method:

align-p2p.cpp: when the realsense camera is connected to a computer, each time, for a captured frame, do a point-to-point mapping from depth image to color image, then get the color image with its pixel-wise corresponding depth information.

data_manipulation.cpp: given the pixel coordinates and depth data of two points (average head and average tail), and the intrinsic matrix, distortion model & coefficients from the camera, calculate the Euclidean distance between these two points, namely the length of a fish.
