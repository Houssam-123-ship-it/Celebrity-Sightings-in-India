# 🧠 Face Detection and Recognition with MTCNN & Inception-ResNet V1

This project demonstrates how to perform **face detection** and **recognition** using deep learning models on a **video of Indian Olympic boxer Mary Kom**.
It combines **MTCNN** (Multi-task Cascaded Convolutional Network) for face detection and **Inception-ResNet V1** for feature extraction and face embedding generation.
Finally, it wraps the entire pipeline into a **Flask web application** that allows users to upload an image and perform real-time face recognition.

---

## 🎯 Project Overview

The project guides you through building a complete end-to-end facial recognition workflow:

1. **Download and process a YouTube video** of Mary Kom’s interview.
2. **Extract frames** from the video for face detection.
3. **Detect and crop faces** using a pre-trained MTCNN model.
4. **Generate high-dimensional embeddings** using Inception-ResNet V1.
5. **Compare embeddings** to recognize known individuals.
6. **Deploy the system** as a Flask web app for user uploads.

---

## 🧩 Learning Outcomes

By completing this project, you’ll learn to:

* Use pre-trained models from **facenet_pytorch** (`MTCNN` and `Inception-ResNet V1`)
* Extract **face bounding boxes** and **facial landmarks**
* Create and visualize **face embeddings**
* Construct a **database of known embeddings**
* Compare new faces against the database using **cosine similarity**
* Build and deploy a **Flask web application**

---

## ⚙️ Step-by-Step Methodology

### 1️⃣ Video Processing and Data Preparation

* Download the **Mary Kom interview video** from YouTube.
* Shorten the clip to one minute to focus on key moments.
* Extract frames at regular intervals for dataset creation.
* Visualize sample frames to verify image quality.

**Concepts Learned**

* Video trimming and frame extraction
* Working with video data in Python using OpenCV

---

### 2️⃣ Face Detection — MTCNN

* Use a **pre-trained MTCNN** model from `facenet_pytorch` to detect faces.
* Identify **bounding boxes** and **facial landmarks** (eyes, nose, mouth).
* Crop aligned faces for consistent feature extraction.

**Key Functions**

```python
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=True)
faces, probs = mtcnn.detect(image)
```

**Output:** cropped faces and detection probabilities.

---

### 3️⃣ Feature Extraction — Inception-ResNet V1

* Each detected face is passed through **Inception-ResNet V1** to obtain a **512-dimensional embedding**.
* These embeddings represent faces as vectors in a **high-dimensional space**, where similar faces cluster together.

#### 🔬 Vectorization Mechanism

* A deep CNN maps an image → vector:
  [
  \text{embedding} = f_{\theta}(x)
  ]
* Similar faces → close vectors (small cosine distance).
* Dissimilar faces → distant vectors.

#### 🔢 Example

```
Mary Kom → [0.31, -0.12, 0.76, …]
Interviewer → [0.07, -0.43, 0.29, …]
```

These embeddings are stored as **reference templates** for recognition.

---

### 4️⃣ Recognition — Cosine Similarity Matching

To identify a face, compare its embedding A to each stored embedding B:

[
\text{Cosine Similarity}(A, B) = \frac{A · B}{‖A‖ × ‖B‖}
]

* Value ≈ 1 → same person
* Value ≈ 0 → different person

If the similarity exceeds a threshold (e.g. 0.8), the system labels the face as **Mary Kom** or **Interviewer**; otherwise → “Unknown”.

---

### 5️⃣ Deployment — Flask Web App

Finally, all code is integrated into a **Flask application** that allows users to:

* Upload an image through a web form.
* Automatically detect faces and perform recognition.
* Display the result with bounding boxes and labels.

**Flask Workflow**

```
User Upload → MTCNN Detection → Embedding Generation → Similarity Comparison → Output Labeled Image
```

This modular architecture keeps the front-end and back-end logic separate for easier testing and scaling.

---

## 🧠 Concepts & New Terms

| Term                  | Definition                                  |
| --------------------- | ------------------------------------------- |
| **Face Detection**    | Locating faces in an image or video frame   |
| **Facial Landmarks**  | Key facial points used for alignment        |
| **Face Embedding**    | Numerical representation of a face          |
| **Cosine Similarity** | Metric for comparing two embeddings         |
| **Flask**             | Python micro-framework for web apps         |
| **Modular Design**    | Code organization using reusable components |

---

## 📁 Repository Structure

```

├── notebooks/
│   ├── 01-video-preprocessing.ipynb
│   ├── 02-face-detection-mtcnn.ipynb
│   ├── NB3/
      ├── app/
      │   ├── static/
      │   ├── templates/
      │   ├── app.py
      |      03-face-recognition-resnet.ipynb
│   ├── NB4/
      |04-flask-deployment.ipynb
│
└── README.md
```

---

## 🧩 Tools & Frameworks

* **Python 3.10+**
* **facenet_pytorch** (MTCNN + Inception-ResNet V1)
* **OpenCV**
* **NumPy**, **Scipy**
* **Matplotlib**
* **Flask**

---

## 🧠 Skills and Competencies Gained

### 💻 Technical Skills

* Deep Learning with PyTorch
* Building end-to-end face recognition systems
* Embedding-based similarity search
* Flask web development and API integration

### 🧠 Analytical & Conceptual Skills

* Vector representation in high-dimensional spaces
* Data preprocessing and model evaluation
* Understanding model architectures (MTCNN, ResNet)

### 💼 Professional Skills

* Modular design & code organization
* Debugging and testing in real-world AI apps
* Ethical and responsible AI practice

---

## 📊 Results and Insights

* Successfully detected and recognized **Mary Kom** and her **interviewer**.
* Achieved **high accuracy** under variable lighting and pose.
* Developed an **interactive Flask app** that performs recognition seamlessly.

---

## 🌍 Ethical Considerations in Computer Vision

This project emphasizes the importance of **ethics** and **responsibility** in AI applications.
Facial recognition technology can have **serious societal implications**—including privacy violations, bias, and misuse.

### 📚 Case Study — Hikvision and Ethnicity Recognition

In *The Guardian* article *“Chinese Security Firm Advertises Ethnicity Recognition Technology While Facing UK Ban”* (Dec 2022), a Chinese company faced allegations of developing racial profiling systems used in surveillance against minority groups.
The report revealed how such technologies can **amplify discrimination** and **violate human rights**.

This case underscores why developers must:

* Prioritize **transparency** and **consent** in data collection
* Avoid creating or supporting **biased datasets**
* Ensure AI systems align with **ethical and legal standards**

**Reference:**
Hern, Alex. “*Chinese Security Firm Advertises Ethnicity Recognition Technology While Facing UK Ban*.” *The Guardian*, 5 Dec 2022.

---

## 📚 References

* facenet_pytorch Documentation
* Parkhi et al. (2015) – *Deep Face Recognition*
* Schroff et al. (2015) – *FaceNet: A Unified Embedding for Face Recognition and Clustering*
* *The Guardian* (2022) Ethical Case Study
* Flask Documentation

---

## 👤 Author

**Houssam Kichchou**
🎓 Data Science & Digital Health Engineering Student
💻 AI and Computer Vision Enthusiast
🌐 [GitHub: Houssam-123-ship-it](https://github.com/Houssam-123-ship-it)

---

Would you like me to add a **diagram (pipeline flow)** showing the full detection → embedding → similarity → recognition process?
It would make the README even more visual and easier to follow.
