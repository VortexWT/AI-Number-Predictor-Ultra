import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('AI (4).keras')


def predict_digit(img):
    img = img.resize((28, 28)).convert('L')  
    img = ImageOps.invert(img)  
    img = np.array(img).reshape(1, 28, 28, 1) / 255.0  
    prediction = model.predict(img)
    return np.argmax(prediction), np.max(prediction)

class App:
    def __init__(self, root):
        self.root = root
        self.pixel_size = 10  
        self.brush_size = 2   
        self.canvas_size = 28
        self.canvas = tk.Canvas(root, width=self.canvas_size * self.pixel_size, height=self.canvas_size * self.pixel_size, bg='white')
        self.canvas.pack()
        
        
        self.canvas.bind("<Button-1>", self.start_paint)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-3>", self.start_erase)
        self.canvas.bind("<B3-Motion>", self.erase)  
        
        
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), 255) 
        self.draw = ImageDraw.Draw(self.image)
        
        self.pixel_map = {}
        self.update_pixel_map()
        
        self.label = tk.Label(root, text="", font=("Helvetica", 24))  
        self.label.pack()
        
        self.update_prediction()

    def update_pixel_map(self):
        """Initialize pixel_map for canvas pixels."""
        for i in range(self.canvas_size):
            for j in range(self.canvas_size):
                x1, y1 = i * self.pixel_size, j * self.pixel_size
                x2, y2 = x1 + self.pixel_size, y1 + self.pixel_size
                self.pixel_map[(i, j)] = (x1, y1, x2, y2)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='gray', fill='white')

    def start_paint(self, event):
        """Start painting on mouse left button press."""
        self.paint(event)
    
    def start_erase(self, event):
        """Start erasing on mouse right button press."""
        self.erase(event)
    
    def paint(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            self.draw_area(x, y, 'black')
    
    def erase(self, event):
        x, y = event.x // self.pixel_size, event.y // self.pixel_size
        if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
            self.draw_area(x, y, 'white')

    def draw_area(self, x, y, color):
        """Draw or erase a 2x2 area starting at (x, y)."""
        for i in range(self.brush_size):
            for j in range(self.brush_size):
                xi, yj = x + i, y + j
                if 0 <= xi < self.canvas_size and 0 <= yj < self.canvas_size:
                    x1, y1, x2, y2 = self.pixel_map[(xi, yj)]
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline='gray', fill=color)
                    self.image.putpixel((xi, yj), 0 if color == 'black' else 255) 

    def update_prediction(self):
        digit, acc = predict_digit(self.image)
        self.label.config(text=f"{digit} ({acc:.2f})")
        self.root.after(100, self.update_prediction) 


root = tk.Tk()
app = App(root)
root.mainloop()
