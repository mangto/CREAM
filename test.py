import cream, time, cv2

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)




while cv2.waitKey(33) < 0:
    start = time.time()
    ret, frame = capture.read()
    frame = cv2.resize(frame, (200, 150))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cream.convolutions(frame, cream.kernel.roberts_1, cream.kernel.roberts_2)
    frame = cream.threshold(frame, 10)
    frame = cv2.resize(frame, (640,480))
    cv2.imshow("VideoFrame", frame)
    print(time.time()-start)

capture.release()
cv2.destroyAllWindows()