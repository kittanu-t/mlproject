# 03_train_evaluate_register.py (สำหรับ Image Classification)
import os
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Global Config (ต้องเหมือนกับ Preprocessing) ---
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42
DATA_DIR = r"C:\mlops_project\archive\Rice_Image_Dataset\Rice_Citation_Request.txt" # <--- ปรับพาธที่นี่ให้ตรงกับที่เก็บไฟล์

def create_cnn_model(input_shape, num_classes):
    """สร้างโมเดล CNN อย่างง่าย"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_evaluate_register(data_dir, epochs=10):
    """
    ฝึกโมเดล CNN, ประเมินผล, และลงทะเบียนโมเดลใน MLflow Model Registry.
    """
    mlflow.set_experiment("Rice Classification - Model Training")

    # 1. โหลดข้อมูลโดยใช้ ImageDataGenerator
    datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=VALIDATION_SPLIT,
        rotation_range=20, # เพิ่ม Data Augmentation
        horizontal_flip=True
    )
    
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        seed=RANDOM_STATE
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=RANDOM_STATE
    )
    
    NUM_CLASSES = train_generator.num_classes
    INPUT_SHAPE = IMAGE_SIZE + (3,) # (128, 128, 3) สำหรับรูปภาพสี

    with mlflow.start_run(run_name=f"cnn_epochs_{epochs}"):
        print(f"Starting CNN training run with {epochs} epochs...")
        mlflow.set_tag("ml.step", "model_training_evaluation")
        mlflow.log_param("epochs", epochs)
        
        # 2. สร้างและฝึกโมเดล CNN
        model = create_cnn_model(INPUT_SHAPE, NUM_CLASSES)
        
        history = model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            steps_per_epoch=train_generator.samples // BATCH_SIZE,
            validation_steps=validation_generator.samples // BATCH_SIZE
        )

        # 3. ประเมินผลโมเดลบน Validation Set
        results = model.evaluate(validation_generator, steps=validation_generator.samples // BATCH_SIZE)
        val_acc = results[1] # accuracy
        
        print(f"Validation Accuracy: {val_acc:.4f}")

        # 4. Log Metrics, และ Model
        mlflow.log_metric("final_val_accuracy", val_acc)
        
        # ใช้ mlflow.tensorflow.log_model สำหรับ Keras/TensorFlow model
        mlflow.tensorflow.log_model(model, "rice_classifier_cnn")

        # 5. ลงทะเบียนโมเดล
        MODEL_NAME = "rice-classifier-prod"
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/rice_classifier_cnn"
        registered_model = mlflow.register_model(model_uri, MODEL_NAME)
        print(f"Model registered as '{registered_model.name}' version {registered_model.version}")

        print("Training run finished.")


if __name__ == "__main__":
    train_evaluate_register(data_dir=DATA_DIR, epochs=5) # อาจปรับ epochs ได้