import cv2
import glob
import numpy as np
import pickle

def get_cals(img_list, db=False):
    #NOTE: Code from CarND-Camera-Calibration reposityory
    #TODO: Cal data where not all corners are visibile is still useable if the objp array is modified accordingly
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    for img_name in img_list:
        img = cv2.imread(img_name)
        ret, corners = cv2.findChessboardCorners(img, (9,6))
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        if db: 
            cv2.imshow("corners", img)
            cv2.waitKey(1)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1280, 720), None, None)
    
    if db: 
        for img_name in img_list:
            img = cv2.imread(img_name)
            cv2.imshow("undist", cv2.addWeighted(img, 0.1, cv2.undistort(img, mtx, dist, None, mtx), 0.9, 0))
            cv2.waitKey(0)

    return {'mtx':mtx, 'dist':dist}

if __name__ == "__main__":
    cal = get_cals(glob.glob("camera_cal/*.jpg"))
    
    with open("camera.cal", "wb") as f:
        pickle.dump(cal, f)

    img = cv2.imread(r"camera_cal\calibration1.jpg")
    und = cv2.undistort(img, cal["mtx"], cal["dist"], None, cal["mtx"])
    cv2.imwrite(r"writeup_files\caltest.jpg", und)
