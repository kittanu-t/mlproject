# 02_data_preprocessing.py (สำหรับ Image Classification)
import os
import mlflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Global Config ---
IMAGE_SIZE = (128, 128) # ขนาดรูปภาพที่ต้องการ
BATCH_SIZE = 32
DATA_DIR = r"C:\mlops_project\archive\Rice_Image_Dataset\Rice_Citation_Request.txt" # <--- ปรับพาธที่นี่ให้ตรงกับที่เก็บไฟล์
RANDOM_STATE = 42

def preprocess_data(data_dir=DATA_DIR, validation_split=0.2, random_state=RANDOM_STATE):
    """
    โหลดข้อมูลรูปภาพ, แบ่งเป็น training/validation, และล็อก class names เป็น artifact.
    """
    mlflow.set_experiment("Rice Classification - Data Preprocessing")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting data preprocessing run with run_id: {run_id}")
        mlflow.set_tag("ml.step", "data_preprocessing")
        mlflow.log_param("validation_split", validation_split)
        mlflow.log_param("image_size", str(IMAGE_SIZE))

        # 1. Initialize ImageDataGenerator for splitting and normalization
        datagen = ImageDataGenerator(
            rescale=1./255, # Normalization (ปรับค่าพิกเซลเป็น 0-1)
            validation_split=validation_split 
        )

        # 2. Load training data generator
        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='training',
            seed=random_state
        )

        # 3. Load validation data generator (สำหรับใช้ในขั้นตอนต่อไป)
        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            subset='validation',
            seed=random_state
        )
        
        # 4. ล็อก Class names เป็น artifact
        class_names = list(train_generator.class_indices.keys())
        
        processed_data_dir = "processed_metadata"
        os.makedirs(processed_data_dir, exist_ok=True)
        
        with open(os.path.join(processed_data_dir, "class_names.txt"), "w") as f:
            f.write("\n".join(class_names))
        
        # 5. ล็อก Metrics และ Artifacts
        mlflow.log_metric("train_samples", train_generator.samples)
        mlflow.log_metric("validation_samples", validation_generator.samples)
        mlflow.log_param("num_classes", train_generator.num_classes)
        mlflow.log_artifacts(processed_data_dir, artifact_path="processed_metadata")
        print(f"Saved class names metadata to MLflow artifacts.")
        
        # คืนค่า run_id และ data_dir เพื่อใช้ในไฟล์ train_evaluate_register
        return run_id, data_dir 

if __name__ == "__main__":
    preprocess_data()