from keras.models import load_model  # TensorFlow is required for Keras to work
import cv2  # Install opencv-python
import numpy as np
from datetime import datetime
import geocoder  # Install geocoder
#tensorflow.keras.models will give error!

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("Resources/Tree_Detector.h5", compile=False)

# Load the labels
class_names = open("Resources/labels.txt", "r").readlines()

# Initialize the camera
camera = cv2.VideoCapture(0)

# Get the latitude and longitude (hardcoded for example or use GPS)
g = geocoder.ip('me')
latitude, longitude = g.latlng

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image_resized)

    # Make the image a numpy array and reshape it to the model's input shape.
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_array = (image_array / 127.5) - 1

    # Predict the model
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print(" Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    # If the detected class is 'tree' (assuming tree has label 0)
    if index == 0:  # index 0 for 'tree', adjust based on your labels
        # Get the current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Text to overlay on the image
        overlay_text = f"Lat: {latitude:.5f}, Lon: {longitude:.5f}, Time: {timestamp}"

        # Position for the text (bottom-left corner of the image)
        position = (10, image.shape[0] - 10)  # 10 pixels from the bottom left corner

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Small size text
        font_color = (0, 255, 0)  # Green color for visibility
        font_thickness = 1

        # Add text to the image
        cv2.putText(image, overlay_text, position, font, font_scale, font_color, font_thickness)

        # Save the image with timestamp, latitude, and longitude in the filename
        filename = f"Tree_{timestamp.replace(':', '-')}_Lat{latitude:.5f}_Lon{longitude:.5f}.jpg"
        cv2.imwrite(filename, image)
        print(f"Tree detected! Image saved as {filename} with overlay text")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()

