import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

filterPicking = True

kernel = np.ones((5,5),np.uint8)
canvas = None
x1,y1=0,0
noise = 200

def ExtractColor(image, mask, default_color=[0,0,255]):
    color = cv2.bitwise_and(image, image, mask=mask)
    mainColor = cv2.mean(color, mask=mask)[:3]

    if np.allclose(mainColor, [0, 0, 0], atol=10):
        return default_color
    print(mainColor)
    return mainColor

def Pass(x):
    pass

def SetupTrackbars(lowerValues, upperValues):
    cv2.namedWindow("Filter Picking")
    cv2.createTrackbar("hue - l", "Filter Picking", lowerValues[0], 179, Pass)
    cv2.createTrackbar("Hue - U", "Filter Picking", upperValues[0], 179, Pass)
    cv2.createTrackbar("saturation - l", "Filter Picking", lowerValues[1], 255, Pass)
    cv2.createTrackbar("Saturation - U", "Filter Picking", upperValues[1], 255, Pass)
    cv2.createTrackbar("value - l", "Filter Picking", lowerValues[2], 255, Pass)
    cv2.createTrackbar("Value - U", "Filter Picking", upperValues[2], 255, Pass)



SetupTrackbars([0, 111, 222], [179, 255, 255])

while(1):
    _, frame = cap.read()
    frame = cv2.flip( frame, 1 )
    if canvas is None:
        canvas = np.zeros_like(frame)

    if filterPicking == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        l_h = cv2.getTrackbarPos("hue - l", "Filter Picking")
        u_h = cv2.getTrackbarPos("Hue - U", "Filter Picking")
        l_s = cv2.getTrackbarPos("saturation - l", "Filter Picking")
        u_s = cv2.getTrackbarPos("Saturation - U", "Filter Picking")
        l_v = cv2.getTrackbarPos("value - l", "Filter Picking")
        u_v = cv2.getTrackbarPos("Value - U", "Filter Picking")

        lowerRange = np.array([l_h, l_s, l_v])
        upperRange = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, lowerRange, upperRange)
    
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        mask_c = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        stacked = np.hstack((mask_c,frame,res))
        
        cv2.imshow('Filter Picking',cv2.resize(stacked,None,fx=0.4,fy=0.4))
        
        if cv2.waitKey(1) == ord('f'):
            filterPicking = False

            lowerRange = np.array([cv2.getTrackbarPos("hue - l", "Filter Picking"),
                                    cv2.getTrackbarPos("saturation - l", "Filter Picking"),
                                    cv2.getTrackbarPos("value - l", "Filter Picking")])
            upperRange = np.array([cv2.getTrackbarPos("Hue - U", "Filter Picking"),
                                    cv2.getTrackbarPos("Saturation - U", "Filter Picking"),
                                    cv2.getTrackbarPos("Value - U", "Filter Picking")])
            cv2.destroyWindow("Filter Picking")
        elif cv2.waitKey(1) == 27:
            break        
    
    else :
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lowerRange, upperRange)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=3)

    
        cont, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cont and cv2.contourArea(max(cont,key = cv2.contourArea)) > noise:
            c = max(cont, key = cv2.contourArea)    
            x2,y2,w,h = cv2.boundingRect(c)
            if x1 == 0 and y1 == 0:
                x1,y1= x2,y2     
            else:
                color = ExtractColor(frame,mask)
                canvas = cv2.line(canvas, (x1,y1),(x2,y2), color, 4)
            x1,y1= x2,y2
        else:
            x1,y1 =0,0

        mask_1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        frame_added = cv2.addWeighted(frame,0.5,canvas,0.5,0)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        stacked = np.hstack((frame_added, res))

        cv2.imshow('Pixionary',cv2.resize(stacked,None,fx=0.6,fy=0.6))
        



        if cv2.waitKey(1) == ord('f'):
            filterPicking = True
            SetupTrackbars(lowerRange, upperRange)
            cv2.destroyWindow("Pixionary")
        elif cv2.waitKey(1) == ord('g') :
            canvas = None

        elif cv2.waitKey(1) == 27:
            break



    if cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()
cap.release()