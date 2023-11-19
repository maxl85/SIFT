import cv2
import numpy as np
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10

# Создаем детектор
detector = cv2.SIFT.create()

# Создаем экстрактор FLANN
FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})

# Находим ключевые точки 
trainImg = cv2.imread("photo.jpg", 0)
trainKP,trainDesc = detector.detectAndCompute(trainImg, None)
trainImg1=cv2.drawKeypoints(trainImg, trainKP, None, (255, 0, 0), 4)
plt.imshow(trainImg1)
plt.show()

# Подключаем захват камеры и уменьшаем размер кадра
cam=cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, QueryImgBGR = cam.read()
    QueryImg = cv2.cvtColor(QueryImgBGR, cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc = detector.detectAndCompute(QueryImg, None)
    matches = flann.knnMatch(queryDesc, trainDesc, k=2)

    goodMatch = []
    for m, n in matches:
        if(m.distance < 0.75*n.distance):
            goodMatch.append(m)
    
    if(len(goodMatch) > MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        
        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape
        trainBorder = np.float32([[[0, 0], [0, h-1], [w-1, h-1], [w-1,0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        
        cv2.polylines(QueryImgBGR, [np.int32(queryBorder)], True, (0, 255, 0), 5)
    else:
        print("Not Enough match found")
    
    cv2.imshow('result', QueryImgBGR)
    
    if cv2.waitKey(10) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()