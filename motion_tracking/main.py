import numpy as np
import cv2


cap = cv2.VideoCapture("Corral 101.mp4")
backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=200)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
result = cv2.VideoWriter('result.mp4',fourcc,
                         60, size)
total = 0
average = 0
max_fish = 0
while(True):
  # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        frame = cv2.medianBlur(frame, 7)
        fgMask = backSub.apply(frame)
        # Our operations on the frame come here


        kernel = np.ones((5, 5), np.uint8)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_ERODE, kernel)
        fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
        contours = cv2.findContours(fgMask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2]
        count =0
        total += 1
        for c in contours:

            x,y,w,h = cv2.boundingRect(c)
            if(w*h >150 and w>20 and h >20):
              count+=1
              cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
              max_fish = max(max_fish,count)
        cv2.putText(frame,'current frame is:'+str(count),(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

        cv2.putText(frame, 'max count is:'+str(max_fish), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

         # Display the resulting frame

        scale_percent = 60  # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
        cv2.imshow('frame',frame)
        # fgMask = cv2.resize(fgMask,dim,interpolation=cv2.INTER_AREA)
        cv2.imshow('fg', fgMask)
        result.write(fgMask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the capture
result.release()
cap.release()
cv2.destroyAllWindows()












# import numpy as np
# import cv2
# import cv2 as cv
# import random as rng
# cap = cv2.VideoCapture('Corral 101.mp4')
#
# if(cap.isOpened() == False):
#     print("error")
#
#
# backSub = cv.createBackgroundSubtractorKNN(dist2Threshold=200)
# while (cap.isOpened()):
#     ret, frame = cap.read()
#     fgMask = backSub.apply(frame)
#
#     kernel = np.ones((3, 3), np.uint8)
#     # fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
#
#     contours, hierarchy = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#     cv.drawContours(fgMask,contours,-1,(0,255,0),20)
#
#
#     if ret == True:
#         cv2.imshow('Frame', frame)
#         cv.imshow('FG Mask', fgMask)
#
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else:
#         break
#
# cap.release()
#
# cv2.destroyAllWindows()
#
#
#
