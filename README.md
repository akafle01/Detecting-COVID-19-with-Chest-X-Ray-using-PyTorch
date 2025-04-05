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


![Screenshot 2025-03-20 at 9 58 43 PM](https://github.com/user-attachments/assets/6ec5d093-994f-4d26-8561-096ad42b7005)
![Screenshot 2025-03-20 at 9 56 14 PM](https://github.com/user-attachments/assets/0f933d1c-0db0-4b45-8aa7-aac17fd690c6)
![Screenshot 2025-03-20 at 9 45 11 PM](https://github.com/user-attachments/assets/3acaf8a3-425d-4b3a-93d5-70827e3caa3f)
![Screenshot 2025-03-20 at 9 40 15 PM](https://github.com/user-attachments/assets/1272dc4a-713e-4ed9-8bce-e2115264818f)
![Screenshot 2025-03-20 at 9 39 53 PM](https://github.com/user-attachments/assets/d431e154-35f5-4ed4-9fac-237705368afd)
<img width="1098" alt="Screenshot 2025-03-03 at 2 36 17 PM" src="https://github.com/user-attachments/assets/f05ccb6a-ce03-480f-b84a-c6973635e36a" />
<img width="850" alt="Screenshot 2025-03-03 at 2 35 22 PM" src="https://github.com/user-attachments/assets/de2891a5-f6e8-4784-9029-999e2a4d7c15" />



🧪 Technologies Used
- Python
- PyTorch
- torchvision
- matplotlib
- PIL
- NumPy
- ResNet-18 (pretrained)
