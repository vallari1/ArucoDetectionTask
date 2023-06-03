import cv2 as cv
from cv2 import aruco
import numpy as np


import cv2 as cv

dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
parameters =  cv.aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)





#markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(framee)
calibration_path = "./calibrate/MultiMatrix.npz"
calibrate = np.load(calibration_path)
print(calibrate.files)
cam_mat = calibrate["camMatrix"]
dist_coef = calibrate["distCoef"]
r_vectors = calibrate["rVector"]
t_vectors =calibrate["tVector"]
MARKER_SIZE = 8  
markerdicty = aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

parameterr = cv.aruco.generateImageMarker(aruco_dict, 10, 50)
capture = cv.VideoCapture(0) #for camera

while True:
    ret, framee = capture.read()
    if not ret:
        break
    grey_frames = cv.cvtColor(framee, cv.COLOR_BGR2GRAY)
    marker_corners, marker_IDs, reject = aruco.detectMarkers(grey_frames, markerdicty, parameterr)
    if marker_corners:
        rVec, tVec, _ = aruco.estimatePoseSingleMarkers(
            marker_corners, MARKER_SIZE, cam_mat, dist_coef)
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv.polylines(framee, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA)
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            distance = np.sqrt(
                tVec[i][0][2] ** 2 + tVec[i][0][0] ** 2 + tVec[i][0][1] ** 2)
            
            point = cv.drawFrameAxes(framee, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)#pose of markers will be given
            cv.putText(
                framee,
                f"id: {ids[0]} Dist: {round(distance, 2)}",
                top_right,
                cv.FONT_HERSHEY_PLAIN,
                1.3,
                (0, 0, 255),
                2,
                cv.LINE_AA,)
            cv.putText(
                framee,
                f"x:{round(tVec[i][0][0],1)} y: {round(tVec[i][0][1],1)} ",
                bottom_right,
                cv.FONT_HERSHEY_PLAIN,
                1.0,
                (0, 0, 255),
                2,
                cv.LINE_AA,)
            
    cv.imshow("frame", framee)
    exit_button = cv.waitKey(1)
    if exit_button == ord("q"):
        break
capture.release()
cv.destroyAllWindows()
