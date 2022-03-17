import cv2 as cv
import numpy as np
import sys
label_name_list = [
    '1WayConnectorforFoley',
    '2WayConnectorforFoley',
    '2WayFoleyCatheter',
    '3WayConnectorforFoley',
    '3Waystopcock',
    'AlcoholBottle',
    'AlcoholPad',
    'BootCover',
    'CottonBall',
    'CottonSwap',
    'Dilator',
    'DisposableInfusionSet',
    'ExtensionTube',
    'FaceShield',
    'FrontLoadSyringe',
    'GauzePad',
    'Glove',
    'GuideWire',
    'LiquidBottle',
    'Mask',
    'NasalCannula',
    'Needle',
    'NGTube',
    'OxygenMask',
    'PharmaceuticalProduct',
    'Pill',
    'PillBottle',
    'PPESuit',
    'PrefilledHumidifier',
    'PressureConnectingTube',
    'ReusableHumidifier',
    'SodiumChlorideBag',
    'SterileHumidifierAdapter',
    'SurgicalBlade',
    'SurgicalCap',
    'SurgicalSuit',
    'Syringe',
    'TrachealTube',
    'UrineBag',
    'Vaccinebottle',
    'WingedInfusionSet',
]

for i,name in enumerate(label_name_list):
    print(f'{i} : {name} \t\t\t',end='')
    if((i+1)%3==0):
        print('') # newline
select_folder_id=0
select_folder_id = int(input('\nEnter Number :'))

if(select_folder_id<0 or select_folder_id>=len(label_name_list)):
    sys.exit()

CONFIDENCE_THRESHOLD = 0.1
NMS_THRESHOLD = 0.4

class_names = []
with open("D:/model-yolo-v4-medicalwaste/obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3)) 

net = cv.dnn.readNet("D:/model-yolo-v4-medicalwaste/yolo-obj_50000.weights", "D:/model-yolo-v4-medicalwaste/yolo-obj.cfg")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

#net = cv.dnn.readNetFromDarknet('D:/model-yolo-v4-medicalwaste/yolo-obj.cfg', 'D:/model-yolo-v4-medicalwaste/yolo-obj_1000.weights')
 
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True, crop=False)

cv.namedWindow('Image', cv.WINDOW_NORMAL)
img_index=1
img_index_changed=True
while(True):
    key = cv.waitKeyEx(40)
    if(key==2424832 or key==2490368 or key==ord('a')): # ←/↑ or a goto previous image 
        print("--")
        img_index-=1
        img_index_changed=True
        if(img_index<1):
            img_index=1
    elif(key==2555904 or key==2621440 or key==ord('d')): # →/↓ or d goto next image
        print("++")
        img_index+=1
        img_index_changed=True
    elif(key==27):
        break

    if(img_index_changed):
        img_index_changed=False
        img = cv.imread('D:/DatasetMedicalWasteTestLabeled/belt/images/belt_'+str(img_index)+'.jpg')
        #img = cv.imread('D:/DatasetMedicalWasteTestLabeled/indoor/'+label_name_list[select_folder_id]+'/'+label_name_list[select_folder_id]+'_'+str(img_index)+'.jpg')
        height_factor = 0.2
        width_factor = 0.2
        img = cv.resize(img,(int(img.shape[0]/height_factor),int(img.shape[1]/width_factor)))
        classes, scores, boxes = model.detect(img, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        for (classid, score, box) in zip(classes, scores, boxes):
                color = COLORS[int(classid)]
                label = "%s : %f" % (class_names[classid[0]], score)
                cv.rectangle(img, box, color, 10)
                cv.putText(img, label, (box[0], box[1]+box[3] - 30), cv.FONT_HERSHEY_SIMPLEX, 4, color, 4)
    cv.imshow('Image', img)


cv.destroyAllWindows()