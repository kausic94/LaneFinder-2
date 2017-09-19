<center><h1>Advanced Lane Finding</h1></center>

___
<p>This document describes the methods I adopted in completing the advanced line finding project. The following files are included.</p>
    
*  Writeup.md - This file, which contains the details about the project.
*  lane_finding.py - python code which has the algorithm implemented
* output.avi - output video that the algorithm has produced
*   output_images - Directory containing the output images

## Step 1: Calibrating the Camera
The first step of the project involved calibrating the camera. Distortion is introduced by the lens in the camera system.This has to be corrected. I was provided with a bunch of chess board images taken at different pose. I used the one in which all the corners can be found inorder to calibrate the camera. This step helped me finding the camera matrix and the distortion coefficients with which I undistorted the image.
<center>
<img src=output_images/calibration1_.jpg></center>
<center>
<i>distorted Image</i>
</center>

This shows one of the distorted images. This image was corrected to the following image after finding the camera matrix and the distortion coefficients.
<center>
<img src=output_images/undistorted_.jpg></center>
<center>
<i> Undistorted Image</i>
</center>

The calibration steps are implemented in the function get_distortion_coeff() which are seen in the lines 176 to 195 in lane_finding.py

##Step 2: Finding Edge Map
<p>
The next step of this process involves finding the edge map of the given camera feed. First I convert the BGR image into HSV space and then I use the S channel to find the edge map. 
<center><img src=output_images/hsv_image_.jpg></center>
<center><i> HSV image </i></center>

To find the edge map I apply a threshold to the channel tha gives me a binary output. This can be seen in the edge_map function(lane_finding.py lines 26-28).
<center><img src=output_images/s_channel_edge_map_.jpg></center>
<center><i> S channel Thresholded </i><center>


I also took the grayscale version of the original image applied a sobel kernel in the x direction to find the edge map. 

<img src=output_images/sobel_x_map_.jpg></center>
<center><i> Sobel X map </i></center>

I combined the two edge maps using an OR operation to get a reliable edge map.I converted it to a 3 channel colored version to visualize it better.The result can be seen below.
<img src=output_images/color_edge_map_.jpg></center>
<center><i> edge map colored </i></center>

I use a binary version of this edge map for further operations. This can be seen implemented in the function get_edge_map() in lane_finding.py lines(14-39).

###Step3: Warping 
In the next step, I warped the image to get a bird's eye view of the road in order to see the lane lines correctly. I applied perspective transform to the images to the get the bird's eye view of the edge map.
<center><img src=output_images/warped_visualised_.jpg></center>
<center><i> bird's eye view of the road</i></center>

This can be seen implemented in the function get_birds_eye_view function in line 41 in lane_finding.py 

###Step4: Finding the Lane line polynomials:
To get the initial lane positions I took the histogram of the lower half of the image of the warped binary image. The peaks of the histogram gave me the initial lane positions.
<center><img src=output_images/histogram_.jpg></center>
<center><i> historgram </i></center>


Once I got the initial position I was able to search an area in a particular window around it and moved up the image to get the intial left and right track points. With the left and right tracks points the corresponding polynomials  were extracted. This can be seen in the following image.
<center><img src=output_images/Visualizing2_.jpg></center>
<center><i> Visualizing the polynomials, left and right tracks </i></center>

This is implemented in the function get_line_position() from line 46 to 100 in the code lane_finding.py

### Step5 : Extracting the polynomials for the next Frames:

 The polynomials once extracted can be used to define the area to be searched in the future frames. From that we can obtain the new track points and thus update the polynomial.  The below image shows this. The green area indicates the area searched.
 <center>!<img src=output_images/lanes_seraching_.jpg></center>
 <center><i> Lane search area</i></center>
 
The function get_lanes_next implements this. It can be seen in the lines 112 to 129 in lane_finding.py

### Step6: Visualizing the lane
The left and right polynomials are used to find the road area. An inverse perspective transform is applied and merged with the original to get the road area. 

### Step 7: Obtaining the original road Parameters:
From the Left and right polynomials the left and the right radius of curvature can be obtained according to the formula given in the classroom. Also, from the assumption that the center of the camera is the center of the lane we can also find out the offset from the center of the road. 
<center><img src=output_images/out_.jpg> </center>

###Conclusion:
The video is processed and saved as a new video namely output.avi. 
