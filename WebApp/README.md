# Object Detection with Roboflow

A Flask web application that uses Roboflow's API to detect objects in images.

## Features

- Upload images (JPG, JPEG, PNG) for object detection
- View detection results with bounding boxes
- Display detailed information about detected objects
- Responsive UI using Bootstrap

## Setup and Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python app.py
```

4. Open your browser and go to `http://127.0.0.1:5000/`

## Environment Variables

You can create a `.env` file in the root directory to store environment variables:

```
SECRET_KEY=your-secret-key
```

## Usage

1. On the home page, click the "Choose an image" button to select an image from your computer
2. Click the "Detect Objects" button to upload the image and analyze it
3. The app will display the original image and the detected objects with bounding boxes
4. Detailed information about the detected objects will be shown below the images

## Dependencies

- Flask
- inference-sdk (Roboflow)
- Python-dotenv
- Pillow

## License
The software is free to use, edit, etc. etc only if the author is given acknowledgment 
