import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
# we are only going to use 4 attributes
COLS = ['Male', 'Asian', 'White', 'Black']
N_UPSCLAE = 1
def extract_features(img_path):
    X_img = face_recognition.load_image_file(img_path)
    locs = face_locations(X_img, number_of_times_to_upsample = N_UPSCLAE)
    if len(locs) == 0:
        return None, None
    face_encodings = face_recognition.face_encodings(X_img, known_face_locations=locs)
    return face_encodings, locs

def predict_one_image(img_path, clf, labels):
    face_encodings, locs = extract_features(img_path)
    if not face_encodings:
        return None, None
    pred = pd.DataFrame(clf.predict_proba(face_encodings),
                        columns = labels)
    pred = pred.loc[:, COLS]
    return pred, locs
def draw_attributes(img_path, df):
    img = cv2.imread(img_path)
    # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    for row in df.iterrows():
        top, right, bottom, left = row[1][4:].astype(int)
        if row[1]['Male'] >= 0.5:
            gender = 'Male'
        else:
            gender = 'Female'

        race = np.argmax(row[1][1:4])
        text_showed = "{} {}".format(race, gender)

        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        # cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        img_width = img.shape[1]
        # font_size = (right - left) / float(img_width) * 30
        # print font_size
        cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    return img



def main():
    output_dir = args.output_dir
    input_dir = args.img_dir
    model_path = args.model

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(model_path) as f:
        clf, labels = pickle.load(f)
    print("classifying images in {}".format(input_dir))
    for fname in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, fname)
        pred, locs = predict_one_image(img_path, clf, labels)
        if not locs:
            continue
        locs = \
            pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
        df = pd.concat([pred, locs], axis=1)
        img = draw_attributes(img_path, df)
        cv2.imwrite(os.path.join(output_dir, fname), img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = \
                        "use pre-trainned resnet model to classify one image")
    parser.add_argument('--img_dir', type=str,
                        default='test', required = True,
                        help='input image directory')
    parser.add_argument('--output_dir', type=str,
                        default='results', help='output directory')
    parser.add_argument('--model', type=str,
                        default='face_model.pkl', required = True,
                        help='path to trained model')
    args = parser.parse_args()
    main()
