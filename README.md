🌿 LeafGuard: AI-Powered Leaf Stress Detection

📌 Overview

LeafGuard is an AI-powered web application that helps farmers and researchers detect stress conditions in plant leaves (such as disease, nutrient deficiency, or environmental stress) using deep learning.
Built with TensorFlow Lite and deployed using Streamlit, LeafGuard is lightweight, fast, and beginner-friendly — making sustainable farming more accessible.

🚀 Features

🌱 AI-based detection of leaf stress using trained deep learning models

📷 Upload images of leaves (JPG/PNG) for instant classification

🧠 TensorFlow Lite model for efficient and portable inference

💻 Streamlit Web App for simple user interaction

🌍 Aligned with UN Sustainable Development Goal #2: Zero Hunger by supporting sustainable agriculture

🧠 Flask used for more efficient inertface

🏗️ Tech Stack
Python 3.11+

Streamlit (Frontend Web App)

TensorFlow Lite (AI Model)

PIL (Pillow) (Image Processing)

NumPy (Data Handling)

📂 Project Structure
leafguard_project/
├── app.py              # Main Streamlit app
├── model.tflite        # Trained TFLite model
├── labels.txt          # Labels for prediction
├── requirements.txt    # Dependencies
└── README.md           # Documentation

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/leafguard.git
cd leafguard

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate  # On Mac/Linux

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run Application
streamlit run app.py


Now open 👉 http://localhost:8501
 in your browser. 🎉

🌐 Deployment

The project can be deployed on:

Streamlit Community Cloud (Free, Easy)

Hugging Face Spaces

Heroku / GCP / AWS

🧪 How It Works

User uploads an image of a leaf.

The app resizes & preprocesses the image.

TensorFlow Lite model classifies the leaf (e.g., Healthy / Diseased / Nutrient Deficient).

The result with confidence score is displayed to the user.
