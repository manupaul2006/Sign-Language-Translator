# Sign Language Dataset Collector 🤟

A Python-based computer vision tool for collecting and organizing **sign language hand-gesture datasets** using a webcam and **MediaPipe Hands**.

This project is intended for collecting training data for a machine-learning system that can later recognize sign language gestures and convert them into text.

## 📌 Overview

The dataset collector uses your webcam to detect hand landmarks in real time.

Instead of manually taking screenshots for every sign, the application can automatically detect a stable hand pose and save it as a training sample.

For every captured sample, the system can store:

* A cropped image of the hand gesture
* Normalized MediaPipe hand landmarks
* Handedness information
* Detection confidence
* Timestamp and other metadata

The collected data can then be used to train a sign-language recognition model.

## ✨ Features

* 🤟 Supports different static hand signs
* 🖐️ Supports up to two hands
* 📷 Automatic image capture
* 🎯 Stable-pose detection
* 🚫 Duplicate-pose prevention
* 🏷️ Sign labeling
* 📊 Sample counter
* 📁 Automatic dataset organization
* 🔢 Saves MediaPipe landmarks as JSON
* 🎮 Manual capture using the keyboard
* 🔄 Reset capture state
* ⚡ Real-time webcam processing

## 🏗️ Project Architecture

```text
                 Webcam
                   │
                   ▼
            OpenCV Video Feed
                   │
                   ▼
             MediaPipe Hands
                   │
          ┌────────┴────────┐
          │                 │
       Hand 1             Hand 2
          │                 │
          └────────┬────────┘
                   ▼
           Landmark Extraction
                   │
                   ▼
             Normalization
                   │
                   ▼
             Stability Check
                   │
                   ▼
           Duplicate Detection
                   │
                   ▼
             Save Sample
             /          \
            ▼            ▼
       Image (.jpg)   Landmarks (.json)
```

## 🛠️ Technologies Used

| Technology | Purpose                                |
| ---------- | -------------------------------------- |
| Python     | Main programming language              |
| OpenCV     | Webcam access and image processing     |
| MediaPipe  | Hand detection and landmark extraction |
| JSON       | Storing landmark data                  |

## 📋 Requirements

Make sure Python is installed on your system.

Install the required packages:

```bash
pip install opencv-python mediapipe
```

The project may also require additional packages depending on the version of the collector you are using.

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository.git
```

Move into the project directory:

```bash
cd your-repository
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the collector:

```bash
python sign_language_dataset_collector.py
```

## 🎮 Controls

| Key     | Action                            |
| ------- | --------------------------------- |
| `A-Z`   | Select the label/sign             |
| `SPACE` | Manually capture the current sign |
| `R`     | Reset duplicate/capture state     |
| `Q`     | Quit the application              |

For example:

```text
Press A
   ↓
Show the sign for A
   ↓
Hold the pose
   ↓
System captures it automatically
```

Then:

```text
Press B
   ↓
Show the sign for B
   ↓
System captures it automatically
```

## 📂 Dataset Structure

The application organizes captured data into separate folders for each sign.

```text
dataset/
│
├── images/
│   ├── A/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   │
│   ├── B/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   │
│   └── C/
│
└── landmarks/
    ├── A/
    │   ├── 0001.json
    │   ├── 0002.json
    │   └── ...
    │
    ├── B/
    │   ├── 0001.json
    │   ├── 0002.json
    │   └── ...
    │
    └── C/
```

## 🧠 Why Save Landmarks?

MediaPipe provides **21 landmarks for each detected hand**.

Instead of relying only on the image, the project also saves the landmark coordinates.

This gives us two possible approaches for the future:

### Image-based recognition

```text
Image
  ↓
CNN / Deep Learning Model
  ↓
Predicted Sign
```

### Landmark-based recognition

```text
Hand Landmarks
      ↓
ML Classifier
      ↓
Predicted Sign
```

Landmark-based models can be much smaller and faster because they work with numerical hand coordinates instead of the complete image.

## 📸 Dataset Collection Guidelines

For a useful machine-learning dataset, avoid collecting every sample from exactly the same position.

Try to collect variations such as:

* Different hand positions
* Different distances from the camera
* Slight rotations
* Different backgrounds
* Different lighting conditions
* Different people, when possible

For example:

```text
A
A at different position
A slightly rotated
A farther from camera
A closer to camera
```

This helps the model learn the **sign itself** instead of memorizing one particular camera setup.

## 👥 Team Workflow

This program is primarily a **development/data-collection tool**.

The workflow for the complete project can be:

```text
            DATA COLLECTION
                   │
                   ▼
        Sign Language Dataset
                   │
                   ▼
             Preprocessing
                   │
                   ▼
           Model Training
                   │
                   ▼
          Trained ML Model
                   │
                   ▼
          Final Application
                   │
                   ▼
        Sign Language → Text
```

The person using the final application does **not** need to manually select labels or press capture keys.

Those controls are for the team while creating the dataset.

## 🔮 Future Improvements

Possible future additions include:

* Real-time sign prediction
* Text output
* Text-to-speech
* Support for larger sign vocabularies
* Dynamic gesture recognition
* Sequence-based models for moving signs
* User-friendly GUI
* Confidence score display
* Dataset statistics
* Automatic train/validation/test splitting
* Model training integration
* Multi-language text output

## ⚠️ Current Limitation

This collector is primarily designed for **static hand gestures**.

Some signs depend on movement over time rather than a single hand pose.

For those signs, a future version should capture a sequence of landmarks:

```text
Frame 1 → Landmark Position
Frame 2 → Landmark Position
Frame 3 → Landmark Position
Frame 4 → Landmark Position
       ↓
    Sequence Model
       ↓
   Sign Prediction
```

Models such as LSTM, GRU, or Transformer-based architectures could eventually be explored for dynamic gestures.

## 🎯 Project Goal

The larger goal of this project is to build a computer-vision-based system capable of recognizing sign language and converting it into readable text.

```text
🤟 Sign Language
       ↓
     Camera
       ↓
   Hand Detection
       ↓
  ML Recognition
       ↓
      Text
```

## 🤝 Contributing

Contributions and improvements are welcome.

You can contribute by improving:

* Data collection
* Preprocessing
* Machine-learning models
* Recognition accuracy
* User interface
* Accessibility
* Dynamic gesture recognition

## 📄 License

Add your preferred open-source license here.

For example:

```text
MIT License
```

## 👨‍💻 Project Status

🚧 **In Development**

Currently focusing on:

* Dataset collection
* Hand landmark extraction
* Sign classification
* Machine-learning model development

The final goal is a working **Sign Language → Text** recognition system.
