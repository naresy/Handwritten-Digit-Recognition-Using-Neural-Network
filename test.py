import numpy as np
import cv2
import pickle
import function  # Your custom module containing `relu` and `softmax`

# Load pre-trained weights
with open('weights.pkl', 'rb') as handle:
    weights = pickle.load(handle, encoding='latin1')

weight1 = weights[0]
bias1 = weights[1]
weight2 = weights[2]
bias2 = weights[3]

# Function to predict the digit
def predict_digit(image, weight1, bias1, weight2, bias2):
    # Flatten and normalize the image
    img_flatten = image.reshape(1, -1).astype('float32') / 255.0
    # Feedforward through the neural network
    input_layer = np.dot(img_flatten, weight1) + bias1
    hidden_layer = function.relu(input_layer)
    scores = np.dot(hidden_layer, weight2) + bias2
    probs = function.softmax(scores)
    return np.argmax(probs)

# Mouse callback for drawing
def draw(event, x, y, flags, param):
    global drawing, canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 10, (255, 255, 255), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# Initialize canvas
canvas = np.zeros((280, 280), dtype='uint8')
drawing = False

cv2.namedWindow('Draw a Digit')
cv2.setMouseCallback('Draw a Digit', draw)

while True:
    cv2.imshow('Draw a Digit', canvas)
    key = cv2.waitKey(1)

    if key == ord('r'):  # Press 'r' to reset the canvas
        canvas.fill(0)

    if key == ord('p'):  # Press 'p' to predict
        # Resize canvas to 28x28 and predict
        resized = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
        prediction = predict_digit(resized, weight1, bias1, weight2, bias2)
        print(f"Predicted Digit: {prediction}")

    if key == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
