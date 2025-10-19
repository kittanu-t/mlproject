# 04_load_and_predict.py (สำหรับ Image Classification)
import mlflow
import numpy as np
from tensorflow.keras.preprocessing import image
import os

# --- Global Config (ต้องเหมือนกับ Training) ---
MODEL_NAME = "rice-classifier-prod"
MODEL_STAGE = "Staging" 
IMAGE_SIZE = (128, 128)
# ลำดับ Class ต้องตรงกับลำดับที่ train_generator สร้างขึ้น
CLASS_NAMES = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag'] 

def load_and_predict(sample_image_path):
    """
    โหลดโมเดล CNN จาก MLflow Model Registry และใช้ทำนายผลบนรูปภาพเดียว
    """
    print(f"Loading model '{MODEL_NAME}' from stage '{MODEL_STAGE}'...")
    
    # โหลดโมเดล
    try:
        # ใช้ mlflow.pyfunc.load_model เพื่อให้โหลด Keras Model ได้
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    except mlflow.exceptions.MlflowException as e:
        print(f"\nError loading model: {e}")
        print(f"Please make sure a model version is in the '{MODEL_STAGE}' stage in the MLflow UI and the training run has completed.")
        return

    # 1. เตรียมรูปภาพตัวอย่างสำหรับการทำนาย
    print(f"Preparing sample image: {sample_image_path}")
    if not os.path.exists(sample_image_path):
         print(f"Error: Sample image file not found at '{sample_image_path}'. Cannot proceed with prediction.")
         return

    # โหลดและปรับขนาดรูปภาพ
    img = image.load_img(sample_image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    # เพิ่มมิติ Batch และทำ Normalization (0-1)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 

    # 2. ใช้โมเดลที่โหลดมาทำนาย
    prediction_proba = model.predict(img_array)[0]
    predicted_class_index = np.argmax(prediction_proba)
    predicted_class_label = CLASS_NAMES[predicted_class_index]
    confidence = prediction_proba[predicted_class_index]

    print("-" * 40)
    print(f"Prediction: {predicted_class_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All class probabilities: {dict(zip(CLASS_NAMES, prediction_proba.round(4)))}")
    print("-" * 40)

if __name__ == "__main__":
    # ***สำคัญ: เปลี่ยนพาธนี้เป็นพาธของรูปภาพข้าวตัวอย่างจริง***
    sample_path = "./Rice_Image_Dataset/Arborio/Arborio (1).jpg" 
    
    if os.path.exists(sample_path):
         load_and_predict(sample_image_path=sample_path)
    else:
         print(f"***Warning: Could not find sample image at {sample_path}***")
         print("Please change the 'sample_path' variable in 04_load_and_predict.py to a valid image file.")