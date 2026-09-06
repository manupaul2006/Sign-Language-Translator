Sign Language Dataset Collector

A simple Python program that uses your webcam to detect hand signs and automatically save them as dataset images and landmark data.

Requirements

- Python 3.9 or newer
- Webcam
- Windows/Linux/macOS

1. Install Python

Make sure Python is installed:

python --version

If Python is not installed, install it from the official Python website.

2. Install Required Libraries

Open a terminal in the project folder and run:

pip install opencv-python mediapipe==0.10.21

3. Run the Program

Run:

python sign_language_dataset_collector.py

Your webcam window should open.

4. Select a Sign Label

Press a letter key to select the sign you want to collect.

For example:

A → Collect sign A
B → Collect sign B
C → Collect sign C

The selected label will be shown on the screen.

5. Capture a Sign

1. Select the label.
2. Show the sign in front of the camera.
3. Keep your hand(s) reasonably still.
4. The program will automatically capture the sign when the pose is stable.

You can also press:

SPACE

to manually capture the current pose.

6. Dataset Location

Captured data is automatically saved inside the "dataset" folder.

The structure will look like:

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

The "images" folders contain the captured hand images.

The "landmarks" folders contain the corresponding MediaPipe hand landmark data in JSON format.

7. Keyboard Controls

Key| Function
"A-Z"| Select sign label
"SPACE"| Manually capture
"R"| Reset capture state
"Q"| Exit program

8. Collecting Multiple Signs

Example:

Press A
↓
Show A sign
↓
Wait for automatic capture
↓
Change the pose slightly if another sample is needed
↓
Press B
↓
Show B sign
↓
Wait for automatic capture

For better training data, collect multiple samples for each sign.

Troubleshooting

Webcam does not open

Make sure:

- Your webcam is connected.
- No other application is currently using it.
- Python has permission to access the webcam.

If you have multiple cameras, you may need to change:

cv2.VideoCapture(0)

to:

cv2.VideoCapture(1)

or another camera index.

MediaPipe installation fails

Try upgrading pip:

python -m pip install --upgrade pip

Then install the dependencies again:

pip install opencv-python mediapipe

Program exits immediately

Run it from the terminal so that any error message remains visible:

python sign_language_dataset_collector.py

Exit

Press:

Q

while the webcam window is active.