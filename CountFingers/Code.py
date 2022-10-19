import cv2
import mediapipe as mp
video = cv2.VideoCapture(0)
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.8,min_tracking_confidence=0.5)
tipid=[8,12,16,20]
def drawhandlandmarks(img,hand_landmarks):
    if hand_landmarks:
        for landmark in hand_landmarks:
            mp_drawing.draw_landmarks(img,landmark,mp_hands.HAND_CONNECTIONS)
def countfingers(img,hand_landmarks):
    if( hand_landmarks):
        landmarks=hand_landmarks[0].landmark
        fingers=[]
        for id in tipid: 
            fingertipy = landmarks[id].y
            fingerbottomy=landmarks[id-2].y
            if(fingertipy<fingerbottomy):
                fingers.append(1)
            else:
                fingers.append(0)
        totalfingers=fingers.count(1)
        text=f'Fingers:{totalfingers}'
        cv2.putText(img,text,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

while True:
    success, img = video.read()
    img=cv2.flip(img,1)
    results=hands.process(img)
    hand_landmarks=results.multi_hand_landmarks
    drawhandlandmarks(img,hand_landmarks)
    countfingers(img,hand_landmarks)
    cv2.imshow('This is your hand',img)
    if cv2.waitKey(1)==32:
        break
cv2.destroyAllWindows()