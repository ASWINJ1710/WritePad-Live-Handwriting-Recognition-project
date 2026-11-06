# ✍️ WritePad – Live Handwriting Recognition

### AI-Powered Handwriting Recognition using Deep Learning and Streamlit

WritePad is a deep learning-based handwriting recognition application that aims to convert handwritten characters, numbers, and symbols into clean digital text in real time.

Users can write on an interactive digital canvas, and the application processes the handwriting using trained deep learning models to generate recognized text.

---

## 🚀 Project Overview

Traditional handwriting recognition systems often require users to upload handwritten images and wait for processing.

WritePad aims to provide a more interactive experience with a digital writing canvas and a separate recognition output panel.

**The main objective is to bridge the gap between handwritten input and digital text using Deep Learning and Computer Vision.**

## ✨ Key Features

- Interactive handwriting canvas.
- Live handwriting recognition workflow.
- Character and digit recognition.
- Support for uppercase and lowercase English characters.
- Number and symbol recognition.
- Image preprocessing for better recognition.
- Deep learning-based character classification.
- Model training and evaluation scripts.
- Separate handwriting input and digital text output.

## 🖥️ Application Interface

The application is designed around two main sections:

| Writing Canvas | Recognition Output |
|---|---|
| User writes characters | Recognized digital text |
| Handwritten input | Clean text output |

*Add your application screenshots here.*

## 🧠 Deep Learning Models

The project explores multiple deep learning architectures for handwritten character recognition.

**Models included in the project:**

- LeNet-5
- AlexNet
- VGGNet
- ResNet
- SpinalNet

**Primary model development:**
- LeNet-based character recognition.
- EMNIST dataset training.
- Fresh model training.
- Fine-tuning and model evaluation.
- Multiple model versions for experimentation.

## 📊 Dataset

**EMNIST (Extended MNIST)**

EMNIST is an extension of the MNIST dataset containing handwritten letters and digits.

The project uses the EMNIST ByClass dataset.

Dataset characteristics:
- 62 character classes.
- 10 digits (0–9).
- 26 uppercase letters (A–Z).
- 26 lowercase letters (a–z).

Dataset reference:

https://www.nist.gov/itl/products-and-services/emnist-dataset

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| PyTorch | Deep learning model development |
| Streamlit | Web application |
| Streamlit Drawable Canvas | Handwriting input |
| NumPy | Numerical operations |
| Pandas | Data handling |
| Pillow | Image processing |
| OpenCV | Computer vision and preprocessing |
| EMNIST | Training dataset |

## 📁 Project Structure

```text
WritePad-Live-Handwriting-Recognition/
│
├── app.py
├── app_backup.py
│
├── models/
│   ├── lenet5.py
│   ├── character_recognition.py
│   ├── content_classifier.py
│   ├── math_recognition.py
│   ├── mixed_recognition.py
│   └── text_recognition.py
│
├── model_hub/
│   ├── fine_tuned/
│   └── pre_trained/
│
├── utils/
│   ├── preprocessing.py
│   └── region_detection.py
│
├── requirements.txt
│
├── train_lenet.py
├── train_lenet_fresh.py
├── train_lenet_v2.py
│
├── evaluate_v2.py
├── evaluate_fresh_model.py
├── evaluate_pretrained.py
│
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ASWINJ1710/WritePad-Live-Handwriting-Recognition-project.git
```

### 2. Navigate to the project directory

```bash
cd WritePad-Live-Handwriting-Recognition-project
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```powershell
.\venv\Scripts\activate
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🏋️ Model Training

To train the LeNet model from scratch:

```bash
python train_lenet_fresh.py
```

To train the updated model:

```bash
python train_lenet_v2.py
```

To evaluate the model:

```bash
python evaluate_v2.py
```

**Note:** Training requires the appropriate EMNIST dataset files. Trained model checkpoints are provided in the `model_hub/` directory.

## 🔬 Model Evaluation

The project includes separate scripts for:

- Model accuracy evaluation.
- Per-character accuracy analysis.
- Pretrained model testing.
- Misclassified sample inspection.
- Model checkpoint verification.
- Recognition testing.

## 🎯 Project Goals

The long-term objective is to develop an intelligent handwriting recognition system capable of:

- Recognizing handwritten characters accurately.
- Identifying numbers and mathematical symbols.
- Converting handwritten input into digital text.
- Supporting mixed handwritten content.
- Improving recognition accuracy through model optimization.
- Providing a smooth real-time writing experience.

## 🚧 Future Improvements

- Improve recognition accuracy for numbers and symbols.
- Implement better word and sentence recognition.
- Improve handwritten character segmentation.
- Explore advanced CNN architectures.
- Add intelligent text correction.
- Improve real-time recognition speed.
- Enhance the user interface and experience.

## 👨‍💻 Author

**Aswin J**

AI / ML Developer | Deep Learning | Computer Vision | Python

GitHub: https://github.com/ASWINJ1710

## 📜 License

This project is intended for educational, research, and portfolio purposes.

---

⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, PyTorch, Deep Learning and Streamlit.**
