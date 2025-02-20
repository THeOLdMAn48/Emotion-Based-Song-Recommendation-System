## JO aa code ma aapde amuk function ni jarur nathi, To pan AAni sarkhi rite samajva mate aapde aano upyog karsu
import cv2 ## cemara sateh live mate
from fer import FER ## ccnn no upyog karva mate
import spotipy ## spotify na geeto no upyog
from spotipy.oauth2 import SpotifyClientCredentials ## spotify ni id no upyog karava
import streamlit as st ## ui desining
import tensorflow
## ama aapde spotify ni credentials set karsu
# of reyon id
my_id = "########### spotify-developer-id #########"
secret = "########### spotify-seceret-key #########"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(my_id,secret))


st.set_page_config(page_title="Music Recommendation", layout="wide")
with st.sidebar:
    st.write("playlists")
    st.write("Setting")
    st.write("About")

st.title("Recommendation")





facecascade = cv2.CascadeClassifier(r"O:\Internship_works ( SEM-5 MACHINE LEARNING )\final_project\haarcascade_frontalface_default.xml")
emotion_detector = FER(mtcnn=True)  ## mtcnn algorithm no upyog karsu aapde ama

# have aapde face detect karava mate 2 function banavsu
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
    genres = genre_list.get(emotion, ['pop'])  ## koi emotion detect na thay to by default pop aave

    ## ama genre etle kaya type no genre and limit etle ketla songs
    result = sp.recommendations(seed_genres=genres, limit=5)
    tracks = result['tracks']

    ##jo aa track chhe e key chhe result dictionry ni mood aa badhu api parj hale che
    track = result['tracks'][0]
    print(result['tracks'])

    ##jo aa track chhe e key chhe result dictionry ni mood aa badhu api parj hale che
    track = result['tracks'][0]
    print(result['tracks'])
    ## have track na artist na name mokalvana
    return track['name'], track['artists'][0]['name']

# def display_songs(img, songs, x, y):
#     for i, song in enumerate(songs):
#         cv2.putText(img, song, (x, y + 30 * (i + 1)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 255, 0), thickness=2)

def draw_boundry(img, classifier, scaleFector, minneighbour, color, text):

    ## aano upyog aapde img ne gray scale ma transfrom karva mate karie chhie
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ## aano upyog jo multiple faces determine karva mate thay che, ke face chhe ke nahi
    features = classifier.detectMultiScale(gray_img, scaleFector, minneighbour)

    ## aa apda courds chhe ke kaya face exist kare che
    cords = []
    for (x, y, w, h) in features:
        face = img[y:y + w, x:x + w]
        ## have aaama aapde imoton detect karsu label maate
        emotions = emotion_detector.detect_emotions(face)
        if emotions:
            dominant_emotion = emotions[0]['emotions']
            emotion_label = max(dominant_emotion, key=dominant_emotion.get)
            ## ractengel etle choras doarava mate !
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            ## aa label vadu uper lakhelu aave e
            cv2.putText(img, emotion_label, (x, y-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=3)

            songs = song_reccomendetion(emotion_label)
            st.write(emotion_label)
            index = 1
            for i in songs:
                st.write(f"{index}.{i}")
                index+=1
            # display_songs(img, songs, x, y + h)

        cords = [x, y, w, h]
    return cords, img

## aano upyog karsu aapde main fucton tarike
def detect(img, facecascade):
    # color = {"blue": (255, 0, 0),
    #          "red": (0, 0, 255),
    #          "green": (0, 255, 0)
    #          }
    # cords, img = draw_boundry(img, facecascade, 1.1, 10, color["green"], "Face")
    # return img

   color = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}
   cords, img = draw_boundry(img, facecascade, 1.1, 5, color["green"], "Face")
   return img


## aa rahiyu ne ema aapda face na data padyo che
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


frame_window = st.image([])
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    img = detect(frame, facecascade)
    frame_window.image(img)
    # cv2.imshow("Frame", img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()
