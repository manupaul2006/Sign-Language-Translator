# Sign Language Dataset Collector

A simple Python program that uses your webcam and MediaPipe to detect hand signs and automatically save them as images and landmark data.

## Requirements

* Python 3.9 or newer
* Webcam
* Windows/Linux/macOS

## Installation

### 1. Install Python

Download and install Python from the official Python website:

https://www.python.org/downloads/

Make sure **Add Python to PATH** is enabled during installation.

### 2. Install the required libraries

Open a terminal in the project folder and run:

```bash
pip install opencv-python mediapipe
```

## Running the Program

Open a terminal inside the project folder and run:

```bash
python sign_language_dataset_collector.py
```

Your webcam window will open.

## How to Use

### Select a Sign

Press a keyboard key from `A` to `Z` to select the label you want to collect.

For example:

```text
Press A → Collect the A sign
Press B → Collect the B sign
Press C → Collect the C sign
```

The selected label will be shown on the screen.

### Automatic Capture

1. Select a label.
2. Show the corresponding sign to the camera.
3. Keep your hand steady for a short moment.
4. The program will automatically capture the sign.

You don't need to press a button for every image.

### Manual Capture

Press:

```text
SPACE
```

to manually capture the current hand pose.

### Reset

Press:

```text
R
```

to reset the capture state if necessary.

### Exit

Press:

```text
Q
```

to close the program.

## Captured Files

The program automatically creates a dataset folder.

Captured images and landmark data are stored separately for each label.

Example:

```text
dataset/
├── images/
│   ├── A/
│   ├── B/
│   └── C/
│
└── landmarks/
    ├── A/
    ├── B/
    └── C/
```

Each captured sign produces:

* `.jpg` — cropped image of the detected hand(s)
* `.json` — MediaPipe landmark information

## Recommended Data Collection

For each sign, collect multiple samples instead of capturing only one image.

While collecting data, vary:

* Hand position
* Distance from the camera
* Slight hand rotation
* Lighting
* Background

This helps create a more useful dataset for training the machine-learning model.

## Troubleshooting

### Camera does not open

Make sure:

* Your webcam is connected.
* No other application is using the webcam.
* You have selected the correct camera in the code if your computer has multiple cameras.

### Hand is not detected

Try:

* Moving closer to the camera.
* Improving the lighting.
* Keeping the entire hand inside the camera frame.
* Avoiding objects covering the hand.

### Program is not responding

Press:

```text
Q
```

to exit and run the program again.

## Controls Summary

| Key     | Action              |
| ------- | ------------------- |
| `A-Z`   | Select sign label   |
| `SPACE` | Manual capture      |
| `R`     | Reset capture state |
| `Q`     | Exit                |
