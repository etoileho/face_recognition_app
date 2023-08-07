import streamlit as st
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
#from google.colab import files
#import os

# 顔のデータをロードする部分
known_face_encodings = []
known_face_names = []

# 新しい顔データの名前を入力
new_face_name = st.sidebar.text_input("覚えさせる名前を入力してください")

# 新しい顔の画像ファイルをアップロード
uploaded = st.sidebar.file_uploader("認証を許可する画像ファイルをアップロードしてください", type=['jpg', 'jpeg', 'png'])
if uploaded:
    # 画像ファイルを保存
    new_face_path = f"{new_face_name}.jpg"
    with open(new_face_path, 'wb') as f:
        f.write(uploaded.read())

    image = face_recognition.load_image_file(new_face_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(new_face_name)
    st.write(f"{new_face_name}のデータが追加されました！最新の顔データのみ認証します。")



st.title('顔認証')

uploaded_file = st.file_uploader("画像ファイルをアップロードしてください", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    face_locations = face_recognition.face_locations(image_array)
    face_encodings = face_recognition.face_encodings(image_array, face_locations)

    im = Image.fromarray(image_array)
    draw = ImageDraw.Draw(im)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "None"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255), width=5)
        draw.text((left + 6, bottom + 6), name, fill=(0, 0, 255, 255))

    st.image(im, caption='認証された顔')
