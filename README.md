🐱🐶 Cats vs Dogs Classification using YOLOv11
This project trains a YOLOv11 Classification model to distinguish between cats and dogs using the classical Cats vs Dogs dataset.
YOLOv11 (2025 release) supports detection, segmentation, pose, and image classification.
In this work, we use the classification version of YOLOv11.

📌 1. YOLOv11 Model Types
YOLOv11 provides several model sizes.
Each size offers a different balance between speed and accuracy:
ModelNameSpeedAccuracyBest forYOLOv11nnano⚡⚡⚡⚡⚡⭐⭐Mobile / Low-power devicesYOLOv11ssmall⚡⚡⚡⚡⭐⭐⭐Fast trainingYOLOv11mmedium⚡⚡⚡⭐⭐⭐⭐Balanced choiceYOLOv11llarge⚡⚡⭐⭐⭐⭐⭐High accuracyYOLOv11xx-large⚡⭐⭐⭐⭐⭐⭐Research / Very high-end GPUs
In this project, yolo11s-cls.pt is used for best performance vs speed.

📂 2. Dataset Structure
Your dataset must follow YOLO's standard classification format:
dataset/
│
├── train/
│   ├── cats/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── dogs/
│       ├── img1.jpg
│       ├── img2.jpg
│       └── ...
│
└── val/
    ├── cats/
    └── dogs/
YOLO automatically uses folder names as class labels.

🔍 3. How YOLO Detects train/ and val/ Automatically
You only need to provide the root folder:
pythondata="/content/drive/MyDrive/dataset/"
YOLO then automatically looks for:

train/
val/
(optional) test/

Classes are inferred from the names of the subfolders inside train/, like:
train/cats  → class 0
train/dogs  → class 1
No YAML file is required for classification training.

🔄 4. Data Augmentation (Automatic)
YOLOv11 applies automatic data augmentation on the train/ folder.
Nothing to configure manually.
Default augmentations include:

Random horizontal flips
Random rotations
Color jitter (brightness, contrast, saturation, hue)
Scaling and cropping
Random blur or noise
Normalization

Validation data is never augmented to ensure accurate evaluation.
To disable augmentation:
pythonaugment=False

🏋️‍♂️ 5. Training Command
pythonfrom ultralytics import YOLO

# Load a pretrained YOLOv11 classification model
model = YOLO('yolo11s-cls.pt')

# Train the model
results = model.train(
    data="/content/drive/MyDrive/dataset/",
    epochs=20,
    imgsz=640,
    batch=32,
    workers=8,
    project="cat_dog_cls",
    name="yolo11"
)

⚙️ 6. Explanation of Training Parameters
ParameterDescriptiondataPath to the dataset containing train/ and val/epochsNumber of full passes through the dataset during trainingimgszThe size YOLO resizes all images to (e.g., 640×640). Your original images can have any sizebatchHow many images are processed at once. Higher = faster but requires more GPU memoryworkersNumber of CPU workers for dataloadingprojectFolder where YOLO logs this experimentnameName of the subfolder inside the project directory
Automatic augmentations are applied only to train/, never to val/.

📈 7. Training Output Files
Once training is done, YOLO creates:
cat_dog_cls/yolo11/
│
├── weights/
│     ├── best.pt     (best validation accuracy)
│     └── last.pt     (last epoch)
│
├── confusion_matrix.png
├── results.png
├── F1_curve.png
└── accuracy.png
These give insights into accuracy, loss curves, and per-class performance.

🧪 8. Inference Example
pythonfrom ultralytics import YOLO

# Load the best trained model
model = YOLO("cat_dog_cls/yolo11/weights/best.pt")

# Run inference on a test image
results = model("test_image.jpg")

# Print results
results[0].show()
Expected output:
dog  (0.97 confidence)

📦 9. Installation
Install the required dependencies:
bashpip install ultralytics

🚀 10. Quick Start

Prepare your dataset following the structure in section 2
Install ultralytics: pip install ultralytics
Run the training script:

pythonfrom ultralytics import YOLO

model = YOLO('yolo11s-cls.pt')
results = model.train(
    data="path/to/dataset/",
    epochs=20,
    imgsz=640,
    batch=32
)

Test your model:

pythonmodel = YOLO("cat_dog_cls/yolo11/weights/best.pt")
model("test_image.jpg")

✅ 11. Conclusion
This project demonstrates how to fine-tune a YOLOv11 classification model using the Cats vs Dogs dataset.
