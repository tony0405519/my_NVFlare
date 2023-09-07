import cv2
dispW=640
dispH=480
flip=2
cam=cv2.VideoCapture(0)
outVid=cv2.VideoWriter('./myCam.avi', cv2.VideoWriter_fourcc(*'XVID'),30,(dispW, dispH))
while True:
    ret, frame= cam.read()
    cv2.imshow('nanoCam', frame)
    outVid.write(frame)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
outVid.release()
cv2.destroyAllWindows()