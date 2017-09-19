#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 00:51:31 2017

@author: kausic
"""

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
# Finding the undistortion matrix.
def edge_map(img):
    global save_images
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV) # get HSV space imaage 
    s_channel=img_hsv[:,:,2]
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # gray scale image    
    x_deriv=cv2.Sobel(gray,cv2.CV_64F,1,0)   # using sobel kernel
    abs_x  =np.absolute(x_deriv)
    x_map=np.uint8(255*abs_x/np.max(abs_x))  #scaling it
    min_thresh,max_thresh=20,100
    x_map_thresholded=np.zeros_like(x_map)
    x_map_thresholded[(x_map>=min_thresh) & (x_map<=max_thresh)]=255 #thresholding it
    
    s_min_thresh,s_max_thresh=170,255
    s_thresholded=np.zeros_like(s_channel)
    s_thresholded[(s_channel>=s_min_thresh)&(s_channel<=s_max_thresh)]=255 # Thresholding S map
    
    color_edge=np.dstack((np.zeros_like(s_thresholded),x_map_thresholded,s_thresholded)) # color 
    combined_binary=np.zeros_like(s_thresholded)
    combined_binary[(x_map_thresholded==255)  | (s_thresholded==255)] = 255  # Color_space and sobel _edge map combined    
    if(save_images):
        cv2.imwrite("output_images/color_edge_map.jpg",color_edge)
        cv2.imwrite("output_image/combined_edge_map.jpg",combined_binary)
        cv2.imwrite("output_image/hsv_image.jpg",img_hsv)
        cv2.imwrite("output_image/s_channel_edge_map.jpg",s_thresholded)
        cv2.imwrite("output_image/sobel_x_map.jpg",x_map_thresholded)
    return combined_binary

def get_birds_eye_view(img,matrix):

    warped_img=cv2.warpPerspective(img,matrix,(img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC)
    return warped_img

def get_line_position(img):
        histogram=np.sum(img[int(img.shape[0]/2):,:],axis=0) #finding histogram
        out_img=np.dstack((img,img,img))                    #converting to a three channel img
        midpoint=np.int(histogram.shape[0]/2)               #finding the midpoint of the histogram

        left_start=np.argmax(histogram[:midpoint])        #left starting point
        right_start=np.argmax(histogram[midpoint:]) + midpoint #right starting point
        
        nwindows=9
        window_ht=np.int(img.shape[0]/nwindows) # setting the sliding window height
        nonzeros=img.nonzero()             # Finding Non-zero pixels
        nonzero_x=np.array(nonzeros[1])
        nonzero_y=np.array(nonzeros[0])
        
        leftx_current=left_start      # setting current centers
        rightx_current=right_start
        margin,minpix=100,50
        left_lane_indx,right_lane_indx=[],[]
        
        for window in range(nwindows):
            # setting up the boundaries for the windows
            win_y_low=img.shape[0]-(window+1)*window_ht
            win_y_high=img.shape[0]-(window)*window_ht
            win_x_left_low=leftx_current - margin
            win_x_left_high=leftx_current + margin
            win_x_right_low=rightx_current-margin
            win_x_right_high=rightx_current+margin
            # Drawing the boundaries
            cv2.rectangle(out_img,(win_x_left_low,win_y_low),(win_x_left_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_x_right_low,win_y_low),(win_x_right_high,win_y_high),(0,255,0), 2) 
            # getting the x and y coordinates of the track within each window
            good_left=((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]
            good_right=((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]
            # shifting the box center if needed
            left_lane_indx.append(good_left)
            right_lane_indx.append(good_right)
            if(len(good_left)>minpix):
                leftx_current = np.int(np.mean(nonzero_x[good_left]))
            if(len(good_right)>minpix):
                rightx_current = np.int(np.mean(nonzero_x[good_right]))
        
        #concatenating the indices into a single array
        left_lane_indx = np.concatenate(left_lane_indx)
        right_lane_indx = np.concatenate(right_lane_indx)
        #finding the x and y coordinates of the left and the right lane
        leftx = nonzero_x[left_lane_indx]
        lefty = nonzero_y[left_lane_indx] 
        rightx = nonzero_x[right_lane_indx]
        righty = nonzero_y[right_lane_indx] 
        #visualizing
        out_img[nonzero_y[left_lane_indx], nonzero_x[left_lane_indx]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_indx], nonzero_x[right_lane_indx]] = [0, 0, 255]
        left_fit=np.polyfit(lefty,leftx,2)
        right_fit=np.polyfit(righty,rightx,2)
        return(out_img,left_fit,right_fit)

def Visualize(img,left_fit,right_fit):
    #plotting left and right lanes seperately for visualization
    plot=np.linspace(0,img.shape[0]-1,img.shape[0])
    poly_fit_line_left=left_fit[0]*plot**2 + left_fit[1]*plot + left_fit[2]
    poly_fit_line_right=right_fit[0]*plot**2 + right_fit[1]*plot + right_fit[2]
    plt.imshow(img)
    plt.plot(poly_fit_line_left,plot,color="yellow")
    plt.plot(poly_fit_line_right,plot,color="yellow")
    plt.show()

def get_lanes_next(img,left_fit,right_fit):
    #finding the lanes from the obtained polynomial
    nonzero=img.nonzero()
    nonzerox=nonzero[1]
    nonzeroy=nonzero[0]
    margin=100
    left_lane=((nonzerox>left_fit[0]*(nonzeroy**2)+left_fit[1]*nonzeroy +left_fit[2] - margin) & (nonzerox<left_fit[0]*(nonzeroy**2)+left_fit[1]*nonzeroy +left_fit[2] + margin))
    right_lane=((nonzerox>right_fit[0]*(nonzeroy**2)+right_fit[1]*nonzeroy + right_fit[2] - margin) & (nonzerox<right_fit[0]*(nonzeroy**2)+right_fit[1]*nonzeroy +right_fit[2]+ margin))
    leftx=nonzerox[left_lane]
    lefty=nonzeroy[left_lane]
    rightx=nonzerox[right_lane]
    righty=nonzeroy[right_lane]
    left_fit=np.polyfit(lefty,leftx,2)
    right_fit=np.polyfit(righty,rightx,2)
    out_img=np.dstack((img,img,img))
    out_img[nonzeroy[left_lane],nonzerox[left_lane]]=[255,0,0]
    out_img[nonzeroy[right_lane],nonzerox[right_lane]]=[0,0,255]
    return (out_img,left_fit,right_fit)
    
def Visualizing_lanes(img,left_fit,right_fit):
    #visualizing the search areas for the lanes.
    margin=100
    plot=np.linspace(0,img.shape[0]-1,img.shape[0])
    leftx=left_fit[0]*plot**2 + left_fit[1]*plot + left_fit[2]
    rightx=right_fit[0]*plot**2 + right_fit[1]*plot + right_fit[2] 
    left_line_window1 = np.array([np.transpose(np.vstack([leftx-margin, plot]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([leftx+margin, plot])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([rightx-margin, plot]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([rightx+margin, plot])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
    cv2.fillPoly(pl_img,np.int32(left_line_pts),[0,255,0])
    cv2.fillPoly(pl_img,np.int32(right_line_pts),[0,255,0])
    img=cv2.addWeighted(img,.8,pl_img,.5,0)
    return img
def get_road(img,left_fit,right_fit,mtx):
    #converting pixel space to world space and obtaining the entire road
    y_factor=30/720
    x_factor=3.7/700
    plot=np.linspace(0,img.shape[0]-1,img.shape[0])
    y_eval=np.max(plot)
    leftx=left_fit[0]*plot**2 + left_fit[1]*plot + left_fit[2]
    rightx=right_fit[0]*plot**2 + right_fit[1]*plot + right_fit[2]
    left_fit_real=np.polyfit(plot*y_factor,leftx*x_factor,2)  # indicates real world polynomial
    right_fit_real=np.polyfit(plot*y_factor,rightx*x_factor,2)
    left_curverad = format(((1 + (2*left_fit_real[0]*y_eval*y_factor + left_fit_real[1])**2)**1.5) / np.absolute(2*left_fit_real[0]),'.2f')
    right_curverad =format(((1 + (2*right_fit_real[0]*y_eval*y_factor + right_fit_real[1])**2)**1.5) / np.absolute(2*right_fit_real[0]),'.2f')
    left_point=left_fit_real[0]*(y_eval*y_factor)**2 +left_fit_real[1]*(y_eval*y_factor) + left_fit_real[2]
    right_point=right_fit_real[0]*(y_eval*y_factor)**2 + right_fit_real[1]*(y_eval*y_factor) + right_fit_real[2]   
    mid_point=(left_point+right_point)/2
    difference=format(x_factor*img.shape[1]/2-mid_point,'.2f') # indicates the offset from the center. negative means the car is tending towards the left annd vice versa
    left= np.array([np.transpose(np.vstack([leftx, plot]))])
    right= np.array([np.flipud(np.transpose(np.vstack([rightx, plot])))])
    road=np.hstack((left,right))
    mask=np.zeros_like(img)
    cv2.fillPoly(mask,np.int32(road),[0,255,0])
    mask=cv2.warpPerspective(mask,mtx,(img.shape[1],img.shape[0]))
    out=cv2.addWeighted(img,.8,mask,.5,0)
    cv2.putText(out,"Left Curvature = "+left_curverad + 'm',(50,50),cv2.FONT_HERSHEY_PLAIN,3,[255,255,2],thickness=2)
    cv2.putText(out,"Right Curvature = "+right_curverad+ 'm',(50,85),cv2.FONT_HERSHEY_PLAIN,3,[255,255,2],thickness=2)
    cv2.putText(out,"Distance from Center = "+difference+ 'm',(50,115),cv2.FONT_HERSHEY_PLAIN,3,[255,255,2],thickness=2)
    return out

save_images=False
def get_distortion_coeff():
    
    calibration_file= "camera_cal/calibration"
    image_pts=[]
    obj_pts=[]
    nx=9
    ny=6
    objp=np.zeros((nx*ny,3),np.float32)
    objp[:,:2]=np.mgrid[0:nx,0:ny].T.reshape(-1,2)
    #calibrate from the given images.
    for i in range(1,21) :
        name=calibration_file + str(i) +'.jpg'
        img=cv2.imread(name,0)
        ret,corners=cv2.findChessboardCorners(img,(nx,ny),None)
        if(ret):
            img=cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
            image_pts.append(corners)
            obj_pts.append(objp)
    ret,mtx,dist,rvecs,tvecs=cv2.calibrateCamera(obj_pts,image_pts,img.shape[::-1],None,None)
    return (mtx,dist)
    
    
if __name__=='__main__':
    
    #Manually get the coordinates for the bird's eye view
    src_pts=np.float32([(240,683),(580,460),(703,460),(1055,683)])
    dst_pts=np.float32([(320,720),(320,0),(980,0),(980,720)])  
    matrix=cv2.getPerspectiveTransform(src_pts,dst_pts)
    matrix_inv=cv2.getPerspectiveTransform(dst_pts,src_pts)
    mtx,dist=get_distortion_coeff()
    # open the video and the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid=cv2.VideoCapture("project_video.mp4")
    if(not vid.isOpened()):
        print("Video Not opened ")
        sys.exit(-1)
    frame=vid.read()[1]
    writer=cv2.VideoWriter('output.avi',fourcc,25,(frame.shape[1],frame.shape[0]))
    #undistort, find edges,warp and get initial line position
    img_undistorted=cv2.undistort(frame,mtx,dist,None,None)
    edge=edge_map(img_undistorted)
    warped=get_birds_eye_view(edge,matrix)
    img,left_fit,right_fit=get_line_position(warped)
    
    #find the lanes for the rest of the frames
    while(True):
        ret,img=vid.read()
        if(not ret):
            print("No Frame")
            break
        img_undistorted=cv2.undistort(img,mtx,dist,None,None)
        warped_original=get_birds_eye_view(img_undistorted,matrix)
        edge=edge_map(img_undistorted)
        warped=get_birds_eye_view(edge,matrix)
        img,left_fit,right_fit=get_lanes_next(warped,left_fit,right_fit)
        pl_img=np.zeros_like(img)
        img=get_road(img_undistorted,left_fit,right_fit,matrix_inv)   
        cv2.namedWindow("img",0)
        cv2.imshow('img',img)
        cv2.waitKey(20)
        writer.write(img)
