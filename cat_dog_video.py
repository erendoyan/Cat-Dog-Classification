from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np

model = load_model("cat_dog_0.83.h5")
cap = cv2.VideoCapture("dog1.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = img.astype(np.float32) / 255.0

    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)

    probability = round(predictions[0][0], 2)
    if probability < 0.5:
        label = "CAT"
    else:
        label = "DOG"

    cv2.putText(frame, f"Prediction: {label} ({probability})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    frame = cv2.resize(frame,(720,480))
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
