import cv2  ## cemara sateh live mate  ## for live interaction with the camera
from fer import FER  ## ccnn no upyog karva mate  ## using CNN for emotion detection
import spotipy  ## spotify na geeto no upyog  ## for fetching songs from Spotify
from spotipy.oauth2 import SpotifyClientCredentials  ## spotify ni id no upyog karava  ## using Spotify credentials
import streamlit as st  ## ui desining  ## for UI design
import tensorflow

## ama aapde spotify ni credentials set karsu  ## setting up Spotify credentials
# of reyon id
my_id = "########### spotify-developer-id #########"
secret = "########### spotify-seceret-key #########"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(my_id, secret))

st.set_page_config(page_title="Music Recommendation", layout="wide")
with st.sidebar:
    st.write("playlists")
    st.write("Setting")
    st.write("About")

st.title("Recommendation")

facecascade = cv2.CascadeClassifier(r"O:\Internship_works ( SEM-5 MACHINE LEARNING )\final_project\haarcascade_frontalface_default.xml")
emotion_detector = FER(mtcnn=True)  ## mtcnn algorithm no upyog karsu aapde ama  ## using MTCNN algorithm

# have aapde face detect karava mate 2 function banavsu  ## now we will create two functions for face detection
def song_reccomendetion(emotion):
    genre_list = {
        'happy': ['pop', 'dance'],
        'sad': ['blues', 'acoustic'],
        'angry': ['metal', 'rock'],
        'neutral': ['classical', 'instrumental'],
        'surprise': ['rock', 'alternative'],
        'fear': ['electronic', 'ambient'],
        'disgust': ['punk', 'indie']
    }
    genres = genre_list.get(emotion, ['pop'])  ## koi emotion detect na thay to by default pop aave  ## default genre is pop if no emotion is detected

    ## ama genre etle kaya type no genre and limit etle ketla songs  ## defining genre type and song limit
    result = sp.recommendations(seed_genres=genres, limit=5)
    tracks = result['tracks']

    ##jo aa track chhe e key chhe result dictionry ni mood aa badhu api parj hale che  ## tracks contain key-value pairs with mood details
    track = result['tracks'][0]
    print(result['tracks'])

    ## have track na artist na name mokalvana  ## sending track name and artist name
    return track['name'], track['artists'][0]['name']

def draw_boundry(img, classifier, scaleFector, minneighbour, color, text):
    ## aano upyog aapde img ne gray scale ma transfrom karva mate karie chhie  ## converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## aano upyog jo multiple faces determine karva mate thay che, ke face chhe ke nahi  ## checking if multiple faces exist
    features = classifier.detectMultiScale(gray_img, scaleFector, minneighbour)

    ## aa apda courds chhe ke kaya face exist kare che  ## coordinates of detected face
    cords = []
    for (x, y, w, h) in features:
        face = img[y:y + w, x:x + w]
        ## have aaama aapde imoton detect karsu label maate  ## detecting emotion and assigning label
        emotions = emotion_detector.detect_emotions(face)
        if emotions:
            dominant_emotion = emotions[0]['emotions']
            emotion_label = max(dominant_emotion, key=dominant_emotion.get)
            ## ractengel etle choras doarava mate !  ## drawing a rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            ## aa label vadu uper lakhelu aave e  ## writing the label above
            cv2.putText(img, emotion_label, (x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=3)

            songs = song_reccomendetion(emotion_label)
            st.write(emotion_label)
            index = 1
            for i in songs:
                st.write(f"{index}.{i}")
                index+=1

        cords = [x, y, w, h]
    return cords, img

## aano upyog karsu aapde main fucton tarike  ## using this as the main function
def detect(img, facecascade):
   color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
   cords, img = draw_boundry(img, facecascade, 1.1, 5, color["green"], "Face")
   return img

## aa rahiyu ne ema aapda face na data padyo che  ## loading face data
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

frame_window = st.image([])
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    img = detect(frame, facecascade)
    frame_window.image(img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
