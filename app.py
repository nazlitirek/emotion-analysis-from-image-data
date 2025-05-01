import cv2
from deepface import DeepFace

# Her duygu için renk (BGR)
emotion_colors = {
    'angry': (0, 0, 255),      # Kırmızı
    'disgust': (0, 255, 0),    # Yeşil
    'fear': (255, 0, 255),     # Mor
    'happy': (0, 255, 255),    # Sarı
    'sad': (255, 0, 0),        # Mavi
    'surprise': (255, 255, 0), # Açık mavi
    'neutral': (128, 128, 128) # Gri
}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'],
            detector_backend='opencv',
            enforce_detection=False
        )

        faces = results if isinstance(results, list) else [results]

        for result in faces:
            region = result["region"]
            emotion = result["dominant_emotion"].lower()
            x, y, w, h = region["x"], region["y"], region["w"], region["h"]

            color = emotion_colors.get(emotion, (255, 255, 255))

            # Dairenin merkezi ve yarıçapı
            center = (x + w // 2, y + h // 2)
            radius = int(max(w, h) * 0.6)

            # Daire çiz
            cv2.circle(frame, center, radius, color, 2)

            # Duygu etiketini merkez üstüne yaz
            cv2.putText(frame, emotion.capitalize(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    except Exception as e:
        print(f"Hata: {e}")

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
