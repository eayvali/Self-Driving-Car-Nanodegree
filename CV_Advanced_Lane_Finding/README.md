## Advanced Lane Finding


The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing the pipeline on single frames. 

# Pipeline
The pipeline was implemented within a class named LaneDetection_From_Video. When the class is initialized, camera calibration is executed. VideoFileClip  takes LaneDetection_From_Video.process_image() as an input  and processes each image sequentially. The processing steps of the pipeline are described in Project_Description.pdf.

![Pipeline Output](advanced_lane_finding.gif)
