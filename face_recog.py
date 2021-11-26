import cv2
import numpy as np
import face_recognition
import os
import pickle

def load_encodings(db_path):
    encodings = []
    names = []
    for file in os.listdir(db_path):
        if(file.endswith(".pickle")):
            with open(os.path.join(db_path,file), "rb") as f:
                encoding = pickle.load(f)
                encodings.append(encoding)
                names.append(file.replace(".pickle",""))
    return encodings, names

class Face_recog():
    def __init__(self,db_path):
        print('Initiate face_recog object')
        self.db_path = db_path
        self.face_encodings_db, self.names_db = load_encodings(db_path)
        print("loaded ", len(self.face_encodings_db), " encodings of ", self.names_db)

    def check_face(self,img):
        """
        Detect a face/ faces in the image
        :param img: 2D array Image
        :return: return a dictionary of { name: (top, right, bottom, left) }
        """
        faces = {}
        if len(self.face_encodings_db) == 0: 
            return faces

        face_locations = face_recognition.face_locations(img)
        encodings = face_recognition.face_encodings(img, face_locations)

        for index, encoding in enumerate(encodings):
            matches = face_recognition.compare_faces(self.face_encodings_db, encoding)
            print(matches)
            face_distances = face_recognition.face_distance(self.face_encodings_db, encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.names_db[best_match_index]
                print('Welcome {}'.format(name))
                (top, right, bottom, left) = face_locations[index]
                if name not in faces:
                    faces[name] = (top, right, bottom, left)
                    print("found")
            else:
                print('unknown')
                return None

        return faces
