# 01_data_validation.py (สำหรับ Image Classification)
import os
import mlflow
import glob

def validate_data(data_dir=r"C:\mlops_project\archive\Rice_Image_Dataset\Rice_Citation_Request.txt"):
    """
    ตรวจสอบโครงสร้างของ Rice Image Dataset และล็อกผลลัพธ์ไปยัง MLflow
    สมมติว่าโครงสร้างคือ: data_dir/[Class_Name]/[Image_File].jpg
    """
    # กำหนดชื่อ Experiment
    mlflow.set_experiment("Rice Classification - Data Validation")

    with mlflow.start_run():
        print("Starting image data validation run...")
        mlflow.set_tag("ml.step", "data_validation")
        mlflow.log_param("data_path", data_dir)

        # 1. ตรวจสอบว่ามี Directory หลักหรือไม่
        if not os.path.isdir(data_dir):
            print(f"Error: Data directory not found at '{data_dir}'")
            mlflow.log_param("validation_status", "Failed: Directory not found")
            return

        # 2. ดึงชื่อ Class (Subdirectories)
        class_names = [d for d in os.listdir(data_dir) 
                       if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
        num_classes = len(class_names)
        
        # 3. นับจำนวนรูปภาพทั้งหมดและต่อคลาส
        total_images = 0
        class_counts = {}
        for class_name in class_names:
            class_path = os.path.join(data_dir, class_name)
            # นับไฟล์ .jpg ในโฟลเดอร์นั้น
            image_files = glob.glob(os.path.join(class_path, "*.jpg"))
            count = len(image_files)
            class_counts[class_name] = count
            total_images += count

        # 4. ล็อกผลลัพธ์
        validation_status = "Success"
        if num_classes < 5 or total_images < 70000: # ชุดข้อมูลมี 5 คลาส, 75000 รูป
            validation_status = "Warning: Low image/class count or incorrect path"
        
        print(f"Number of classes found: {num_classes}")
        print(f"Total images found: {total_images}")
        print(f"Validation Status: {validation_status}")
        
        mlflow.log_metric("num_classes", num_classes)
        mlflow.log_metric("total_images", total_images)
        mlflow.log_param("validation_status", validation_status)
        mlflow.log_text(str(class_names), "class_names.txt")

        print("Data validation run finished.")

if __name__ == "__main__":
    validate_data(data_dir=r"C:\Users\ACE\Desktop\mlops\archive\Rice_Image_Dataset") # ปรับพาธที่นี่