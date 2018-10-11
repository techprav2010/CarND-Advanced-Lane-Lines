##  **Advanced Lane Finding**

**Finding Lane Lines on the Road**

* Steps taken to detect lane lines.  
    * Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    * Apply a distortion correction to raw images.
    * Use color transforms, gradients, etc., to create a thresholded binary image.
    * Apply a perspective transform to rectify binary image ("birds-eye view").
    * Detect lane pixels and fit to find the lane boundary.
    * Determine the curvature of the lane and vehicle position with respect to center.
    * Warp the detected lane boundaries back onto the original image.
    * Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[calibration18_jpg_original]: ./output_images/calibration18_jpg_original.png
[calibration18_jpg_undistort]: ./output_images/calibration18_jpg_undistort.png 
[calibration18_jpg_calibration_original_ChessboardCorners]: ./output_images/calibration18_jpg_calibration_original_ChessboardCorners.png

[straight_lines2_jpg_original]: ./output_images/straight_lines2_jpg_original.png 
[straight_lines2_jpg_undistort]: ./output_images/straight_lines2_jpg_undistort.png 

[straight_lines1_jpg_original_image]: ./output_images/straight_lines1_jpg_original_image.png 
[straight_lines1_jpg_perspective_transformed]: ./output_images/straight_lines1_jpg_perspective_transformed.png 
[test1_jpg_RGB2HLS_perspective]: ./output_images/test1_jpg_RGB2HLS_perspective.png 
[test1_jpg_RGB2HLS_L_Channel_perspective]: ./output_images/test1_jpg_RGB2HLS_L_Channel_perspective.png 
[test1_jpg_RGB_perspective]: ./output_images/test1_jpg_RGB_perspective.png 
[test1_jpg_RGB2HLS_S_Channel_perspective]: ./output_images/test1_jpg_RGB2HLS_S_Channel_perspective.png 


[test1_jpg_RGB_R_Channel_perspective]: ./output_images/test1_jpg_RGB_R_Channel_perspective.png 
[test1_jpg_color_channel_RGB2Lab_perspective]: ./output_images/test1_jpg_color_channel_RGB2Lab_perspective.png 
[test1_jpg_color_channel_RGB2HLS_S_Channel_perspective]: ./output_images/test1_jpg_color_channel_RGB2HLS_S_Channel_perspective.png 
[test1_jpg_RGB2HLS_S_Channel_perspective]: ./output_images/test1_jpg_RGB2HLS_S_Channel_perspective.png 
[test1_jpg_RGB_perspective]: ./output_images/test1_jpg_RGB_perspective.png 

 

[straight_lines2_jpg_draw_curve_original_image]: ./output_images/straight_lines2_jpg_draw_curve_original_image.png 

[test6_jpg_draw_lane_filtered_image]: ./output_images/test6_jpg_draw_lane_filtered_image.png 
[test6_jpg_filter_and_gradient_bin_img_sobel_abs_x]: ./output_images/test6_jpg_filter_and_gradient_bin_img_sobel_abs_x.png 
[test6_jpg_filter_and_gradient_bin_img_thresh_sobel_dir]: ./output_images/test6_jpg_filter_and_gradient_bin_img_thresh_sobel_dir.png 
[test6_jpg_filter_and_gradient_bin_img_thresh_sobel_mag]: ./output_images/test6_jpg_filter_and_gradient_bin_img_thresh_sobel_mag.png 
[test6_jpg_filter_and_gradient_binary_thresh_rgb_b]: ./output_images/test6_jpg_filter_and_gradient_binary_thresh_rgb_b.png 
[test6_jpg_final_perspective_original_preview]: ./output_images/test6_jpg_final_perspective_original_preview.png 


[test6_jpg_draw_curve_img_windows_1]: ./output_images/test6_jpg_draw_curve_img_windows_1.png 
[test6_jpg_draw_curve_original_image]: ./output_images/test6_jpg_draw_curve_original_image.png 

[straight_lines1_jpg_draw_rect_histogram]: ./output_images/straight_lines1_jpg_draw_rect_histogram.png 
[straight_lines1_jpg_draw_rect_img_windows_1]: ./output_images/straight_lines1_jpg_draw_rect_img_windows_1.png 
[straight_lines1_jpg_draw_rect_original_image]: ./output_images/straight_lines1_jpg_draw_rect_original_image.png 
[straight_lines1_jpg_draw_curve_filtered_image]: ./output_images/straight_lines1_jpg_draw_curve_filtered_image.png 

[straight_lines1_jpg_draw_lane_img_windows_1]: ./output_images/straight_lines1_jpg_draw_lane_img_windows_1.png  


---

### Reflection

#### Compute the camera calibration matrix and distortion coefficients given a set of chessboard images. And apply distortion correction to raw images.
   
* Camera calibration and distortion correction:
    * Camera (lenses) distort images from real world (3D) to image (2D). 
    * To correct image distortion we need to calculate transition matrix. It can be reverse engineered by finding distortion in known image samples.
    * A chess board can be used get such sample images to measure distortions. Once we have find the camera_calibration_matrix​, which maps the
distorted points to undistorted points,  we can apply on images taken by that camera, And undo the distortion.   
 * AlCalibration class : Wrapper for OpenCV methods for camera calibration utilities. 
    * OpenCV provides methods like - findChessboardCorners() and drawChessboardCorners().
    * Samples: There are 21 chessboard image provided in 'camera_cal' folder. Chessboard grid has 9 X 6 dimension.
    * OpenCV does the calculation and returns: distortion_coefficients​, camera_calibration_matrix​. 
    * The distortion introduced by camera can be now corrected using - distortion_coefficients​, camera_calibration_matrix​.
 * images
    * Original image (chessboard)
    - ![alt text][calibration18_jpg_original]
    * Convert image in o undistorted using calibration (chessboard)
    - ![alt text][calibration18_jpg_undistort]
    * Corners during calibration (chessboard)
    - ![alt text][calibration18_jpg_calibration_original_ChessboardCorners]
     * Original image
    - ![alt text][straight_lines2_jpg_original]
    * Convert image in to undistorted using calibration
    - ![alt text][straight_lines2_jpg_undistort]
    
#### Apply a perspective transform to rectify binary image ("birds-eye view").
* Perspective transformation: Birds-eye
    * First un-distort the image using camera_calibration_matrix​ and distortion_coefficients​.
    * Choose three points in source and three points in to be warped image (birds-eye perspective)
    * OpenCV utility methods for transformation - getPerspectiveTransform, warpPerspective
    * Once you have chosen points, its time to transform/warp image into birds-eye perspective.
* images
    * straight_lines1.jpg
    - ![alt text][straight_lines1_jpg_original_image]
    * straight_lines1.jpg perspective_transformed
    - ![alt text][straight_lines1_jpg_perspective_transformed]
    * test1_jpg  RGB_perspective
    - ![alt text][test1_jpg_RGB_perspective]
    * test1_jpg RGB2HLS_perspective
    - ![alt text][test1_jpg_RGB2HLS_perspective]
    * test1_jpg RGB2Lab_perspective
    - ![alt text][test1_jpg_color_channel_RGB2Lab_perspective]
     * test1_jpg RGB2HLS_S_Channel_perspective
    - ![alt text][test1_jpg_RGB2HLS_S_Channel_perspective] 
     * test1_jpg RGB_R_Channel_perspective
    - ![alt text][test1_jpg_RGB_R_Channel_perspective]
   * test1_jpg RGB2HLS_L_Channel_perspective
    - ![alt text][test1_jpg_RGB2HLS_L_Channel_perspective]
  

     
#### Use color transforms, gradients, etc., to create a thresholded binary image.
* Identify the useful pixels for lane detection:
    * First warp undistorted-image into birds-eye perpective image, use region of interest to reduce the noise.
    * Filter pixels by applying thresholds on various colors channels (GRAY, RBB2HLS, RGB2LAB, BGR2RGB etc).
        * It will be useful to consider yellow and white colors and gray version.
    * Filter pixels by applying thresholds on derivatives (derivatives using sobel operator). 
        * Gradient in x
        * Gradient magnitude : x**2 + y**2 
        * Absolute values: abs(x)
    * Once you find useful pixels for lane detection, create a binary image using those points.
 
* images        
    * test6_jpg  
    - ![alt text][test6_jpg_final_perspective_original_preview]
    * test6_jpg filter_and_gradient_bin_img_sobel_abs_x
    - ![alt text][test6_jpg_filter_and_gradient_bin_img_sobel_abs_x]
    * test6_jpg filter_and_gradient_bin_img_thresh_sobel_mag
    - ![alt text][test6_jpg_filter_and_gradient_bin_img_thresh_sobel_mag]
    * test6_jpg filter_and_gradient_bin_img_thresh_sobel_dir
    - ![alt text][test6_jpg_filter_and_gradient_bin_img_thresh_sobel_dir]
    * test6_jpg filter_binary_thresh_rgb_b
    - ![alt text][test6_jpg_filter_and_gradient_binary_thresh_rgb_b]
    * test6_jpg filtered_image
    - ![alt text][test6_jpg_draw_lane_filtered_image]

      
#### Determine the curvature of the lane and vehicle position with respect to center. and  warp the detected lane boundaries back onto the original image.
* Polynomial fitting, measuring curvature and vehicle position:
    * We have a binary image with useful lane pixel (co-ordinates) identified.
    * Take histograms and divide it into left and right lane points.
    * Use sliding window algorithm to identify the change in density of points (in near small neighborhood).
    * Find a polynomial to fit those points as summarized above. ( x = y^2 + By + C )
    * From window to window the line may change slowly. 
    * if points not available in next window, continue using line detected in prev window. 
 
* images
    * straight_lines1_jpg_draw_rect_original_image
    - ![alt text][straight_lines1_jpg_draw_rect_original_image]
    * straight_lines1_jpg_draw_curve_filtered_image
    - ![alt text][straight_lines1_jpg_draw_curve_filtered_image]
    * straight_lines1_jpg_draw_rect_histogram
    - ![alt text][straight_lines1_jpg_draw_rect_histogram]
    * straight_lines1_jpg_draw_rect_img_windows_1
    - ![alt text][straight_lines1_jpg_draw_rect_img_windows_1]
    * test6_jpg_draw_curve_img_windows_1
    - ![alt text][test6_jpg_draw_curve_img_windows_1] 

#### Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
* Lane curvature and vehicle position:
    * Draw the fitted line on real image, now we are considering the change using small sliding windows.
    * Pixels into meters : Since our camera has fixed position and direction, we can convert pixels into meter. e.g. { y_m_per_pix = 10ft = 3.048m , x_m_per_pix = 12ft = 3.7m}
    
* images
    * straight_lines1_jpg draw_lane_img_windows_1
    - ![alt text][straight_lines1_jpg_draw_lane_img_windows_1]
    
        
        
####  Problems encountered
* It needs more work to decide 'points of interest', specifically on steep curves.
* Should do better job on thresholding colors and gradients to reduce noise. 
    * e.g.
        * shadows
        * vehicle in front obstructing view
        * missing lanes
        * etc
* Poly fit cloud be improved in noisy situation or when data is lost in previous sliding window.
        
 
 

 
