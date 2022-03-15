import cv2
 
#img = cv2.imread('D:/DatasetMedicalWasteTestLabeled/belt/images/belt_138.jpg')
img = cv2.imread('D:/DatasetMedicalWasteTestLabeled/outdoor/OxygenMask/OxygenMask_7.jpg')

CONFIDENCE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)] 

class_names = []
with open("D:/model-yolo-v4-medicalwaste/obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("D:/model-yolo-v4-medicalwaste/yolo-obj_50000.weights", "D:/model-yolo-v4-medicalwaste/yolo-obj.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

#net = cv2.dnn.readNetFromDarknet('D:/model-yolo-v4-medicalwaste/yolo-obj.cfg', 'D:/model-yolo-v4-medicalwaste/yolo-obj_1000.weights')
 
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
 
classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
 
for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid[0]], score)
        cv2.rectangle(img, box, color, 2)
        cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()