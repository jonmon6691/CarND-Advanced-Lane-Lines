# Writeup

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal_img]: ./camera_cal/calibration1.jpg
[undist_cal_img]: ./writeup_files/caltest.jpg

[dist_road]: output_images\straight_lines2_Step00_input.jpg
[undist_road]: ./writeup_files/straight_lines2_Step01_undistorted.jpg

[undist_yellow]: ./writeup_files/straight_lines1_Step01_undistorted.jpg
[yellows]: ./writeup_files/straight_lines1_Step02_yellows.jpg
[whites]: ./writeup_files/straight_lines1_Step03_whites.jpg
[thresholded]: ./writeup_files/straight_lines1_Step04_color_mask.jpg

[persp_src]: ./writeup_files/persp_src.jpg
[persp_dst]: ./writeup_files/persp_dst.jpg

[cm_warp]: writeup_files\test4_Step05_cm_warp.jpg
[lf]: writeup_files\test4_Step06_lane_finder.jpg
[lf_close]: writeup_files\test4_Step06_lane_finder_zoom.jpg

[final]: writeup_files\test2_Step08_final.jpg

[video1]: ./project_video.mp4

## Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

## Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file called `calibrate_camera.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. This would have to change if for some reason differnt targets were used to calibrate the image within the dataset. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

| Input Image | Undistorted Image |
| --- | --- |
| ![cal_img] | ![undist_cal_img] |

The calibration matrix and distortion coefficients are then stored in a pickle file so that they can be loaded when executing the pipeline. This avoids having to recalculate them every time since they will not change.

```python
    cal = get_cals(glob.glob("camera_cal/*.jpg")) 
    with open("camera.cal", "wb") as f:
        pickle.dump(cal, f)
```


## Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The calibration matrix and distortion coefficients are loaded from a python pickle when the pipeline program starts. This prevents them from having to be recalculated while the pipeline is running. This significantly improves performance on video, but also helps keep the code,test,debug loop tight.

```python
with open("camera.cal", "rb") as f:
    cal = pickle.load(f)
```

The Open CV call `cv2.undistort` is simply called with the loaded constants generated in `calibrate_camera.py`

```python
img = cv2.undistort(img, mtx, dist, None, mtx)
```

| Input Image | Undistorted Image |
| --- | --- |
| ![dist_road] | ![undist_road] |

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color thresholds to generate a binary image (thresholding steps at lines # through # in `pipeline.py`). The goal in this step was to have the lines represented clearly with as little noise near them as possible, and to do so in under a wide variety of lighting conditions and road quality.

To achieve this, I combined two parallel pipelines, one that selected yellow lines, and another that selected white lines.

```python
    # Color space conversions
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h = hls[:,:,0]
    s = hls[:,:,1]
    l = hls[:,:,2]

    # Find yellow lines
    ret, yellow = cv2.threshold(h, 17, 255, cv2.THRESH_BINARY) # h > 17
    ret, yellow2 = cv2.threshold(h, 34, 255, cv2.THRESH_BINARY_INV) # h < 34
    yellows = cv2.bitwise_and(yellow, yellow2) # 17 < h < 34
    ret, l = cv2.threshold(l, 90, 255, cv2.THRESH_BINARY) # l > 90
    yellows = cv2.bitwise_and(yellows, l)

    # Find white lines
    ret, whites = cv2.threshold(s, 200, 255, cv2.THRESH_BINARY) # s > 200

    # Combine color masks
    color_mask =  cv2.bitwise_or(yellows, whites)
```

| `img` | `yellows` | `whites` | `color_mask` |
| ----- | --------- | -------- | ------------ |
| ![undist_yellow] | ![yellows] | ![whites] | ![thresholded] |

Note, I didn't end up using the Sobel operator to find gradients because I had trouble eliminating noise in the final image.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The first step was to define the transform source and destination points. These were hand-picked off the clearest example image to make a trapezoid with parallel top and bottom lines, and side lines aligned on the lane lines. The assumption is that the lines should become parallel in the un-warped image. The destination points and image size were chosen to preserve as much resolution near the car as possible, and allowing for adequate curvature in either direction.

```python
persp_src = np.float32([[285, 664], [1012, 664], [685, 450], [596, 450]])
td_width = 1280
td_height = 1280
persp_dst = np.float32([[ll_targ, td_height-10], [rl_targ, td_height-10], [rl_targ, td_height//4], [ll_targ, td_height//4]])
M = cv2.getPerspectiveTransform(persp_src, persp_dst)
```

| Source points visualized | Perspective shifted image |
| ----- | --------- |
| ![persp_src] | ![persp_dst] |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane finding algorithm I designed works pixel row by pixel row from the bottom up on the perspective shifted, threshold image from the previous two steps. The lane state is captured in the `LaneFinder` class located in `lanes.py` and the row processing is done in the loop at line ## in `pipeline.py`

| Threshold image after perspective xform | Debug output from lane finding algorithm | Closeup |
| ----- | --------- | --- |
| ![cm_warp] | ![lf] | ![lf_close]

A triangular window is convoluted with the image row and the max value of that convolution within a search window is selected as the center of the lane for that row. The windows are initialized to constant positions at the bottom of the image. As the search progresses upward, the windows are moved based on three things: the distance to the other line, the line of best fit of the points discovered so far, and the noise detected in the window. This trajectory based approach allows the window to find dashed lines effectively.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `pipeline.py`

Once the lane finding is finished, the found points along the line are scaled into meters. The scaling factors were hand calculated using the assumption that a lane in the test images was 3.7m wide and a dash in a dashed line was 3m long, and measuring the corresponding pixel lengths.

```python
rad_l = np.sqrt((1 + (2 * l.fit2[0] * y_eval + l.fit2[1])**2)**3) / (2 * l.fit2[0]) 
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `pipeline.py`.

The lanes are painted up until the last detected pixel, and the green region is painted up until the last point both lanes had a pixel. This gives a visual indication of the confidence of the lane detection, and will still show something when only a single line was found. 

![final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The frames in the video had to be converted to BGR, processed in the pipeline, then converted back to RGB. This is primarily because the yellow lane selection filter depends on the correct hue value on the input.

Here's a [link to my video result](writeup_files\project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

This was an interesting project because of the interaction between the different pieces of the pipeline. I started by carefully crafting each piece in order trying to maximize the performance of each block before moving onto the next. This proved to be a poor way to manage my time, so I shifted to implementing a simple version of each block so that as I went back to improve them, I could see the impact throughout the pipeline.
The approach I used was tuned well for the project video but performs quite poorly on the two challenge videos. This is due to the assumptions made about the visibility and apparent color of the lines. In the project video the light is uniform and bright and the lanes are clear and solid. In the two challenge videos, they are difficult to see even with your own eyes, and in some cases are completely invisible due to the lighting condition or field of view of the camera. I did not consider the previous frame when processing video and that could have improved the speed and accuracy of the search for the lane in subsequent frames.

