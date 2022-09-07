import cv2

threshold=600

def cluster_analysis(frame,i):
    connectivity = 8
    output = cv2.connectedComponentsWithStats(frame, connectivity, cv2.CV_32S)
    stats = output[2]
    k=0
    lis=[]
    for l in stats:
        if l[4]>=threshold:
            lis.append([l[0],l[1],l[0]+l[2], l[1]+l[3]])
    k+=1
    return lis

cap = cv2.VideoCapture('example.mp4')
ret, frame = cap.read()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(20,20))

fgbg = cv2.createBackgroundSubtractorMOG2()
i=0

while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)

    ret, img = cv2.threshold(fgmask, 127, 255, 0)

    lis = cluster_analysis(img,i)
    for l in lis:
        frame = cv2.rectangle(frame,(l[0],l[1]),(l[2],l[3]),(0,255,0),3)

    #sayları koseye yaz
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,str(len(lis)-1),(60,60), font, 2,(255,255,255))
    #frame = cv2.fillPoly( frame, pts, 0 )

    #print frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) == ord('q'):
        break
    i+=1

cap.release()
cv2.destroyAllWindows()
