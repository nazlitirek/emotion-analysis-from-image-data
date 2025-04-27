import cv2
from fastai.vision.all import *

# Kaydedilen modeli yükle
learn = load_learner('model.pkl')

# Kamera başlatma
cap = cv2.VideoCapture(0)

# Kamera açıldıysa
if not cap.isOpened():
    print("Kamera açılamadı!")
else:
    print("Kamera başarıyla açıldı!")

# Kamera görüntüsünü almak için
ret, frame = cap.read()

# Görüntüyü işlemek
if ret:
    # OpenCV'nin BGR formatında olduğu için, bunu RGB formatına çevirmemiz lazım
    img = PILImage.create(frame)  # frame OpenCV formatında alınan görüntü

    # Modelin tahmin yapması
    pred_class, pred_idx, outputs = learn.predict(img)
    print(f'Predicted class: {pred_class}, Index: {pred_idx}')
    print(f'Outputs (probabilities): {outputs}')

# Kamera kapatma
cap.release()
cv2.destroyAllWindows()
