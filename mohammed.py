import time
import cv2
import mediapipe  as mp
import matplotlib.pyplot as plt
import numpy as np
import os

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
plt.show()

def mediapipe_detection(image, model):
        # Converting from the BGR format to RGB, because the mediapipe library deals with the RGB format, unlike the Open Computer Vision library, which deals with the BGR format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = model.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results


def draw_landmarks(image, results):
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)  # Draw face connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
        # Draw face connections
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                                )
        # Draw pose connections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                                )
        # Draw left hand connections
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                                )
        # Draw right hand connections
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                )



def extract_keypoints(results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

"""result_test = extract_keypoints(results)
    print(result_test)
    np.save('0', result_test)
    np.load('0.npy')
    """
DATA_PATH = os.path.join(r"E:\sign_language\MP_Data\\")

actions = np.array(["hello","thanks","happy"])#"happy","again","thanks","hello",'what'

    # Thirty videos worth of data
    #no_sequences = 30

    # Videos are going to be 30 frames in length
sequence_length = 30

    # Folder start
start_folder = 0


    #step 6. Preprocess Data and Create Labels and Features
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(actions)}
sequences, labels = [], []
for action in actions:
        for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
            window = []
            for frame_num in range(sequence_length):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
"""X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)"""


    #step 7  Build and Train LSTM Neural Network
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

    #model.add(Dense(actions.shape[0], activation='softmax'))
    #model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    #model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])
"""
    model.save('action.h5')
    del model"""
model.load_weights('action.h5')

    #step 10
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


    # 11. Test in Real Time
from scipy import stats

colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]


"""def prob_viz(res, actions, input_frame, colors):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), colors[num], -1)
            cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_frame"""

    # 1. New detection variables
def fromsignlang():
    actions = np.array(["hello ,","thanks ,","i love you ,"])
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5


    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                #res1 = model.predict(np.expand_dims(sequence[-5:], axis=0))[0]

                predictions.append(np.argmax(res))
                #lis.append(actions[np.argmax(res)])
                #lis2.append(actions[np.argmax(res1)])



                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('h'):
                break
        cap.release()
        cv2.destroyAllWindows()
    return(sentence)








words=['clean', 'hearing', 'intent', 'learn', 'like_love', 'meet', 'name', 'no', 'sign', 'slow', 'slowly', 'student', 'teacher', 'what', 'where', 'who', 'why', 'yes', 'your']
def frommoviepy(sentences):
    videos=[]

    from moviepy.editor import VideoFileClip, concatenate_videoclips
    sentences=sentences.replace("?","") 
    if ","  not in sentences:
         bm=0
         print(45)
    else :
         bm= sentences.count(",")
    for i in range(bm +1 ):
        
        if sentences.lower() == "are you student" :
            videos.append(os.path.join(r"E:\sign_language\data\\",("are you student"+".mp4")))
        
        elif (bm) in ["what's your name", "what is your name"]:
            videos.append(os.path.join(r"E:\sign_language\data\\",("what your name"+".mp4")))
        
        
        elif  ( sentences.lower() in ["don't" , "donot" , "didnot" , "didn't"]) and (sentences.lower in  ["grasp" , "understand" , "comprehend" , "gotit" ] ):
            videos.append(os.path.join(r"E:\sign_language\data",("don't understand"+".mp4")))
        
        
        elif (sentences.lower()) == ("are you deaf"):
            videos.append(os.path.join(r"E:\sign_language\data\\",("deaf you"+".mp4")))
        
        elif sentences.lower() in [ "your", "yours", "belongtoyou" ,"toyou"] :
            videos.append(os.path.join(r"E:\sign_language\data\\",("your"+".mp4")))
        
        
        else :
            sentences=sentences+" "
            for i in range(sentences.count(" ")+1):
                bm=sentences.index(" ")
                word=sentences[:bm]
                if word in words:
                    videos.append(os.path.join(r"E:\sign_language\data",word+".mp4"))
                sentence=sentences[:bm+1]

    if len(videos) == 1:
        video_file = str(videos[0])
        path = video_file
    else:
        
   
        final_video= concatenate_videoclips(videos, method="compose")
        final_video.write_videofile(r"C:\Users\ASUS\Downloads\sign_language\result.mp4")
        path =r"C:\Users\ASUS\Downloads\sign_language\result.mp4"
    return path


def show(path):
    cap = cv2.VideoCapture(path)

    # التحقق مما إذا كان يمكن فتح ملف الفيديو بنجاح
    if not cap.isOpened():
        pass
    else:
        while True:
            # قراءة إطار من ملف الفيديو
            ret, frame = cap.read()

            # التحقق مما إذا تم قراءة الإطار بنجاح
            if not ret:
                break

            # عرض الإطار
            cv2.imshow('Video', frame)

            # انتظار لحظة قبل قراءة الإطار التالي (قد تكون هذه القيمة مختلفة اعتمادًا على سرعة الإطار في ملف الفيديو)
            cv2.waitKey(25)

    # إغلاق ملف الفيديو وإغلاق نوافذ العرض
    cap.release()
    cv2.destroyAllWindows()


import openai

# استبدال هذه السلسلة بمفتاح API الخاص بك
api_key = "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"

# نص المحادثة الأولي
conversation = []

def API(user_input):

    user_input =  "Connect the following words to form a meaningful sentence : " + user_input
    conversation.append({"role": "user", "content": user_input})
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        api_key=api_key
    )

    bot_reply = response["choices"][0]["message"]["content"]

    conversation.append({"role": "assistant", "content": bot_reply})
    
    return (bot_reply)

    

from gtts import gTTS
import os

# النص الذي تريد تحويله إلى صوت

# تحويل النص إلى صوت باللغة العربية

x=""
while x != "exite":
    x =  str(input("what you're thinking about : "))
    if x != "exite":

        frommoviepy(x)


        time.sleep(3)

        bm = fromsignlang()

        print(bm)
        bm= ", ".join(bm)

        bm=str(bm)

        AIDF =API(bm)

        tts = gTTS(text=AIDF , lang='en')

        # حفظ الصوت في ملف
        tts.save("output.mp3")

        # تشغيل الملف الصوتي
        os.system("start output.mp3")
