import cv2
from fastai.vision.all import *

# Load the trained model
learn = load_learner('model.pkl')

# Open the webcam (index 0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the image to a PIL image (required by FastAI)
    pil_img = PILImage.create(img)

    # Make prediction
    pred_class, pred_idx, outputs = learn.predict(pil_img)

    # Display the prediction on the frame
    cv2.putText(frame, f'Prediction: {pred_class}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with prediction
    cv2.imshow("Emotion Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
