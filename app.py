import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load the pre-trained TensorFlow model for digit recognition
model = tf.keras.models.load_model("BEST-AI 2.0.keras")

def predict_digit(img):
    """
    Preprocesses the image and makes a prediction using the trained model.
    
    Args:
        img (PIL.Image.Image): The image to predict.
        
    Returns:
        np.ndarray: The model's prediction probabilities.
    """
    img = img.resize((28, 28)).convert("L")  # Resize image to 28x28 and convert to grayscale
    img = ImageOps.invert(img)  # Invert image colors (black becomes white and vice versa)
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0  # Normalize and reshape image for model input
    prediction = model.predict(img)  # Predict digit probabilities
    return prediction

class App:
    def __init__(self, root):
        """
        Initializes the main application window and its components.
        
        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.attributes("-fullscreen", True)  # Set window to fullscreen
        self.root.configure(bg="#121a22")  # Dark grey background

        # Bind the Escape key to the exit function
        self.root.bind("<Escape>", self.exit_app)

        # Define canvas parameters
        self.pixel_size = 25
        self.brush_size = 3
        self.canvas_size = 28

        # Create frame for the canvas and set grid configuration
        self.canvas_frame = tk.Frame(root, bg="#121a22")
        self.canvas_frame.grid(row=0, column=1, sticky="nsew")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # Create the canvas widget
        self.canvas = tk.Canvas(
            self.canvas_frame,
            width=self.canvas_size * self.pixel_size,
            height=self.canvas_size * self.pixel_size,
            bg="white",  # Colour  canvas
            highlightthickness=0
        )
        self.canvas.pack(expand=True)

        # Bind mouse events to canvas functions
        self.canvas.bind("<Button-1>", self.start_paint)  # Left click to start painting
        self.canvas.bind("<B1-Motion>", self.paint)  # Hold and move to paint
        self.canvas.bind("<Button-3>", self.clear_canvas)  # Right click to clear canvas

        # Initialize image and drawing objects
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # White background image
        self.draw = ImageDraw.Draw(self.image)  # Drawing object

        # Create frame for prediction labels
        self.label_frame = tk.Frame(root, bg="#121a22")  # Dark grey background
        self.label_frame.grid(row=0, column=2, sticky="nsew")

        # Create labels to display prediction probabilities
        self.labels = [
            tk.Label(
                self.label_frame,
                text="",
                font=("Helvetica", 20, "bold"),
                fg="#d7e5d8",
                bg="#121a22"
            )
            for _ in range(10)
        ]
        for label in self.labels:
            label.pack()

        # Create label to display the predicted digit and confidence
        self.result_label = tk.Label(
            root,
            text="",
            font=("OCR-A BT", 36, "bold"),
            fg="#d7e5d8",
            bg="#547808"
        )
        self.result_label.grid(row=1, column=1, sticky="nsew")

        # Start the prediction update loop
        self.update_prediction()

    def start_paint(self, event):
        """Start painting on mouse left button press."""
        self.paint(event)

    def paint(self, event):
        """
        Paints on the canvas at the location of the mouse event.
        
        Args:
            event (tk.Event): The mouse event.
        """
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            self.draw_area(x, y, "black")  # Draw a black square

    def draw_area(self, x, y, color):
        """
        Draw or erase a 2x2 area starting at (x, y).
        
        Args:
            x (int): X-coordinate of the starting position.
            y (int): Y-coordinate of the starting position.
            color (str): Color to draw ("black" or "white").
        """
        for i in range(self.brush_size):
            for j in range(self.brush_size):
                xi, yj = x + i, y + j
                if 0 <= xi < self.canvas_size and 0 <= yj < self.canvas_size:
                    x1, y1 = xi * self.pixel_size, yj * self.pixel_size
                    x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="", fill=color)
                    self.image.putpixel((xi, yj), 0 if color == "black" else 255)

    def clear_canvas(self, event):
        """Clear the canvas on right-click."""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255)  # Reset to white background
        self.draw = ImageDraw.Draw(self.image)  # Reinitialize drawing object

    def update_prediction(self):
        """
        Update the prediction labels and result label based on the current canvas image.
        """
        predictions = predict_digit(self.image)
        print(f"Predictions shape: {predictions.shape}")  # Debug statement to check prediction shape

        # Ensure predictions have the expected shape
        if len(predictions.shape) == 2 and predictions.shape[1] == 10:
            for i, label in enumerate(self.labels):
                percentage = predictions[0][i] * 100
                # Format percentage, ensuring 100.0% displays correctly
                formatted_percentage = "100.0%" if percentage == 100.0 else f"{percentage:.2f}%"
                label.config(text=f"{i}: {formatted_percentage}")

            predicted_digit = np.argmax(predictions)
            confidence = np.max(predictions) * 100
            # Format confidence, ensuring 100.0% displays correctly
            formatted_confidence = "100.0%" if confidence == 100.0 else f"{confidence:.2f}%"
            self.result_label.config(
                text=f"Predicted: {predicted_digit} ({formatted_confidence})"
            )
        else:
            # Handle errors in prediction shape
            for label in self.labels:
                label.config(text="Error in prediction shape")
            self.result_label.config(text="Error")

        # Schedule the next update
        self.root.after(100, self.update_prediction)

    def exit_app(self, event):
        """Exit the application when the Escape key is pressed."""
        self.root.quit()

# Create and run the Tkinter application
root = tk.Tk()
app = App(root)
root.mainloop()
