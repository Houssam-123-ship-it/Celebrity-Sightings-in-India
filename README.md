Here’s your **README.md** file content ready to save and use:

---

```markdown
# Face Detection and Recognition with Mary Kom Video

This project demonstrates **face detection and recognition** using a video of Indian Olympic boxer **Mary Kom**. It combines state-of-the-art computer vision models with a user-friendly **Flask web application** to allow face recognition on new images.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Ethics in Computer Vision](#ethics-in-computer-vision)
- [References](#references)

---

## Overview
In this project, you will:

- Download a YouTube interview video of Mary Kom.
- Shorten the video and extract frames for processing.
- Detect faces using **MTCNN** (Multi-task Cascaded Convolutional Network).
- Create **face embeddings** with **Inception-ResNet V1**.
- Recognize faces from a library of known embeddings.
- Wrap the functionality into a **Flask web app** for easy user interaction.

---

## Features

### 1. Video Processing
- Download YouTube video.  
- Trim the video to required length.  
- Extract frames at regular intervals.  

### 2. Face Detection
- Use pre-trained **MTCNN** model from `facenet_pytorch`.  
- Detect faces and facial landmarks (eyes, nose, mouth).  
- Crop detected faces for embedding.  

### 3. Face Recognition
- Generate **face embeddings** with **Inception-ResNet V1**.  
- Compare embeddings to identify known faces.  
- Label faces in new images based on embedding similarity.  

### 4. Flask Web App
- Upload an image for face recognition.  
- Display output image with bounding boxes and labels.  
- Modular design: recognition logic separated from app interface.  

---

## Project Structure
```

face_recognition_project/
│
├── app.py              # Flask web app entry point
├── face_recognition.py # Face detection and recognition functions
├── embeddings.pt       # Stored face embeddings
├── upload.html         # Upload interface
├── requirements.txt    # Python dependencies
├── notebooks/          # Jupyter notebooks for exploration
│   ├── 042-data-exploration.ipynb
│   ├── 043-face-detection-mtcnn.ipynb
│   ├── 044-face-recognition-inceptionresnet.ipynb
│   └── 045-flask-api.ipynb
└── data/
├── video.mp4       # Input video of Mary Kom
└── frames/         # Extracted frames from video

````

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Houssam-123-ship-it/Celebrity-Sightings-in-India
````

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure `ffmpeg` is installed for video processing:

   ```bash
   ffmpeg -version
   ```

---

## Usage

### Step 1: Extract Frames

```bash
python face_recognition.py --extract_frames
```

> Or run `043-face-detection-mtcnn.ipynb` to explore frames interactively.

### Step 2: Generate Embeddings

```bash
python face_recognition.py --generate_embeddings
```

> Alternatively, use `044-face-recognition-inceptionresnet.ipynb` to compute embeddings.
> The output is saved as `embeddings.pt`.

### Step 3: Run Flask App

```bash
python app.py
```

* Open your browser at: `http://127.0.0.1:5000`
* Upload an image via `upload.html`
* View recognized faces with bounding boxes and labels.

---

## Technologies Used

* **Python 3.x**
* **PyTorch**
* **facenet_pytorch**
* **OpenCV**
* **Flask**
* **HTML/CSS**

---

## Ethics in Computer Vision

Building face recognition systems requires ethical awareness:

* Respect **user privacy** and data protection.
* Avoid **bias** and **discrimination** in datasets or models.
* Obtain consent when collecting or analyzing facial data.
* Do not use recognition for **intrusive surveillance**.

**Reference:**

> Hern, Alex. “Chinese Security Firm Advertises Ethnicity Recognition Technology While Facing UK Ban.” *The Guardian*, 2022.
> [Read here](https://www.theguardian.com/world/2022/dec/04/chinese-security-firm-advertises-ethnicity-recognition-technology-while-facing-uk-ban)

---

## References

1. [facenet_pytorch Documentation](https://github.com/timesler/facenet-pytorch)
2. [Inception-ResNet V1 Paper](https://arxiv.org/abs/1602.07261)
3. [Flask Documentation](https://flask.palletsprojects.com/)

---

### Author

[**Houssam Kichchou**](www.linkedin.com/in/houssam-kichchou)
Data Science & Digital Health Engineering Student
Supervised by [**World Quant University**](https://www.wqu.edu/)

```

