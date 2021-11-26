import cv2
import os
import time
import pickle
import numpy as np
import face_recognition



if(__name__ == "__main__"):
    cap = cv2.VideoCapture('/dev/video0')
    name = input("Enter your name: ")
    img_path = f'registered_img/{name}/'

    feature_path = "registered_feature"
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)

    status = True
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if os.path.exists(img_path):
        status = False

    count = 0

    while(status):
        _, frame = cap.read()
        width = cap.get(3)
        height = cap.get(4)

        plain_frame = frame.copy()

        cv2.rectangle(frame, (int(width/3) , int(height/6)), (int(width*2/3), int(height*5/6)), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "Press 's' to register", (int(width/3), int(height*5/6) + 30), font, 1.0, (0, 0, 255), 2)

        cv2.imshow("video", frame)

        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break
        elif pressedKey == ord('s'):
            img_save_path=img_path+'image'+str(count)+'.jpg'
            cv2.imwrite(img_save_path,plain_frame)
            time.sleep(2)
            print('image'+str(count)+' saved')
            count+=1
        
        if count == 5:
            break

    cap.release()
    cv2.destroyAllWindows()

    img_list = [m for m in os.listdir(f'registered_img/{name}')]
    
    extracted_feat = []
    for img_name in img_list:
        img = cv2.imread(img_path+img_name)
        feat = face_recognition.face_encodings(img)[0]
        extracted_feat.append(feat)

    extracted_feat = np.asarray(extracted_feat)
    mean_feature = np.mean(extracted_feat.reshape(len(img_list),128), axis=0)

    with open(os.path.join(feature_path, name+".pickle"), "wb") as f:
        pickle.dump(mean_feature, f)

    print(f"{name} feature added to {feature_path}")        
            
    