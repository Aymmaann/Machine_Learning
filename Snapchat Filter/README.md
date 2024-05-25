# Snapchat Filter Project

This project aims to create a real-time face filter application inspired by Snapchat, utilizing Python's OpenCV library.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/Aymmaann/Machine_Learning/tree/main/Snapchat%20Filter
    ```

2. Install the required dependencies:

    ```bash
    pip install opencv-python numpy pandas
    ```

3. Download the required XML files for face and eye detection:

    - `frontalEyes35x16.xml`: [Download](https://github.com/Aymmaann/Machine_Learning/blob/main/Snapchat%20Filter/Train/third-party/frontalEyes35x16.xml)
    - `Nose18x15.xml`: [Download](https://github.com/Aymmaann/Machine_Learning/blob/main/Snapchat%20Filter/Train/third-party/Nose18x15.xml)

4. Add the XML files to the project directory.

## Usage

1. Modify the `image_url` variable in the script to specify the path of the input image.
2. Run the script:

    ```bash
    python snapchat.py
    ```

3. The modified image with applied filters will be displayed. Press any key to close the window.
4. A CSV file containing the RGB pixel values of the modified image will be saved in the `Modified_Images` directory.

*Feel free to experiment with different filters and effects to create your unique face filters!*
