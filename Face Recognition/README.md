# Facial Recognition Project

This repository contains a facial recognition project implemented using OpenCV and a K-Nearest Neighbors (KNN) algorithm. The project includes two main scripts:
1. `face_data_collect.py` - For collecting face data and saving it.
2. `face_recognition.py` - For recognizing faces using the collected data.

## Project Structure

- `data/`: Directory to store the collected face data.
- `face_data_collect.py`: Script to collect face data and save it as `.npy` files.
- `face_recognition.py`: Script to recognize faces using the KNN algorithm.
- `haarcascade_frontalface_alt.xml`: Haar Cascade file for face detection.
- `README.md`: Project documentation.

## How It Works

### Face Data Collection (`face_data_collect.py`)

This script uses a webcam to capture images of a person's face, detects the face using a Haar Cascade classifier, and saves the processed face data into a `.npy` file. 

**Steps:**
1. Initialize the webcam.
2. Load the Haar Cascade classifier for face detection.
3. Capture frames from the webcam.
4. Convert frames to grayscale and detect faces.
5. Extract and resize the face region, then save it to a list.
6. When done, save the collected face data to a `.npy` file in the `data/` directory.

### Face Recognition (`face_recognition.py`)

This script uses the KNN algorithm to recognize faces from the webcam feed. It loads the previously collected face data, detects faces in the current frame, and then predicts the identity of the detected face using KNN.

**Steps:**
1. Load the Haar Cascade classifier for face detection.
2. Load the saved face data from the `data/` directory.
3. Capture frames from the webcam.
4. Convert frames to grayscale and detect faces.
5. Extract and resize the face region.
6. Use the KNN algorithm to predict the identity of the face.
7. Display the predicted identity on the frame.

## Usage

### Prerequisites

- Python 3.x
- OpenCV
- NumPy

Install the required libraries:

```bash
pip install opencv-python numpy
```


## Collecting Face Data
1. Ensure the Haar Cascade file is located at haarcascade_frontalface_alt.xml.
2. Run the face data collection script:
```bash
python face_data_collect.py
```
3. Enter the name of the person when prompted.
4. The script will start the webcam, detect faces, and save the face data to the data/ directory.
5. Press q to stop the data collection.


## Recognizing Faces
1. Ensure the Haar Cascade file and face data files are in the correct locations.
2. Run the face recognition script:
```bash
python face_recognition.py
```
3. The script will start the webcam, detect faces, and recognize them using the KNN algorithm.
4. Press q to stop the face recognition.

## Notes
- Modify the dataset_path and Haar Cascade file path in the scripts if needed.
- Ensure your webcam is properly connected and accessible by OpenCV.

## Troubleshooting
- If the Haar Cascade file cannot be loaded, check the file path and ensure the file exists.
- If the webcam is not working, ensure it is properly connected and accessible in your system settings.
- For issues with NumPy or OpenCV, ensure the libraries are correctly installed using pip.

## Contributing
Contributions are welcome! Please fork the repository and create a pull request with your changes.