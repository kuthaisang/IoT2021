import cv2
import face_recog
import face_recognition
import time

def main():
    model = face_recog.Face_recog("./registered_feature")
    cap = cv2.VideoCapture("/dev/video0")
    while(True):
        _,frame = cap.read()
        currentTime = time.process_time()
        names = model.check_face(frame)
        print(names)
        for name in names:
            top, right, bottom, left = names[name]
            frame = cv2.rectangle(frame, (left,top), (right,bottom), (0, 0, 255), 5)
            frame = cv2.putText(frame, name, (left, bottom+30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 3)

        frame = cv2.putText(frame, "FPS: " + str(round(1/(time.process_time() - currentTime))), (10,80), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("recognize", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows







if(__name__ == "__main__"):
    main()