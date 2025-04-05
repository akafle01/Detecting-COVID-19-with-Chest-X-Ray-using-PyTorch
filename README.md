# 🦠 Detecting COVID-19 with Chest X-Ray using PyTorch

This project demonstrates the use of deep learning for image classification on chest X-ray images to detect **COVID-19**, **Viral Pneumonia**, and **Normal** conditions using **PyTorch** and **ResNet-18**.

> 🧠 Guided by the Coursera project: [Detecting COVID-19 with Chest X Ray using PyTorch](https://www.coursera.org/projects/covid-19-detection-x-ray)  
> 📦 Dataset from Kaggle: [COVID-19 Radiography Dataset](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

---

## 📌 Project Overview

The goal of this project is to train a convolutional neural network to classify chest X-ray images into three categories:
- **Normal**
- **Viral Pneumonia**
- **COVID-19**

Using **transfer learning** with ResNet-18, I aim to achieve high accuracy on a held-out test set. The final model reaches over **95% accuracy** after just one epoch.

---

## 🗂️ Sections

### 1. 🧾 Introduction
This project is a deep learning-based image classification task using PyTorch. The model is trained to identify whether a chest X-ray shows signs of:
- Normal lungs
- Viral Pneumonia
- COVID-19

---

### 2. 🗃️ Preparing and Splitting the Dataset
I used the *COVID-19 Radiography Dataset* from Kaggle. The dataset was reorganized to separate training and testing samples. For each class, 30 random images were moved to a test folder.
`python
source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
class_names = ['normal', 'viral', 'covid']
# Create test folders and move images

### 3. 🧰 Creating Custom Dataset
A custom ChestXRayDataset class was built using PyTorch’s Dataset abstraction. It loads images dynamically from the labeled folders.
class ChestXRayDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        ...

### 4. 🖼️ Image Transformations
Image preprocessing and augmentation included resizing, normalization, and horizontal flipping (for training only).

train_transform = transforms.Compose([...])
test_transform = transforms.Compose([...])

### 5. 🧪 Prepare DataLoaders
The dataset was wrapped using PyTorch DataLoader for batch processing and shuffling.

dl_train = DataLoader(train_dataset, batch_size=6, shuffle=True)
dl_test = DataLoader(test_dataset, batch_size=6, shuffle=True)

### 6. 🔍 Data Visualization
To evaluate prediction quality, a custom function displays images with true and predicted labels. Correct predictions are shown in green; incorrect ones in red.

def show_images(images, labels, preds):
    ...

### 7. 🧠 Creating the Model
Used ResNet-18 with pretrained weights. The final fully connected layer was modified to output 3 classes.

resnet18 = torchvision.models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(512, 3)
We used:

Loss Function: CrossEntropyLoss

Optimizer: Adam (learning rate = 3e-5)

### 8. 🚂 Training the Model
Training was done for 1 epoch with early stopping if accuracy exceeded 95%. Periodic evaluations on the test set were performed during training.

def train(epochs=1):
    ...

### 9. ✅ Final Results
📈 Validation Accuracy Over Time:

Step 0: 27.78%

Step 20: 84.44%

Step 40: 88.89%

Step 60: 95.56%

🧠 The model hit the accuracy target and training stopped early.

🧪 Technologies Used
- Python
- PyTorch
- torchvision
- matplotlib
- PIL
- NumPy
- ResNet-18 (pretrained)
