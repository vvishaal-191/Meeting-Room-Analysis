import tkinter as tk
from tkinter import filedialog
from tkinter import Label
import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('Age_Sex_Detection.keras')

class AgeGenderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Age and Gender Detection")
        
        self.label = Label(root, text="Upload an Image")
        self.label.pack()
        
        self.upload_button = tk.Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()
        
        self.result_label = Label(root, text="")
        self.result_label.pack()
        
        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()
        
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.display_image(file_path)
            self.predict_image(file_path)
    
    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((400, 400), Image.LANCZOS)
        self.img = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img)
    
    def predict_image(self, file_path):
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        image = np.expand_dims(image, axis=0) / 255.0
        
        pred = model.predict(image)
        
        age = int(np.round(pred[1][0][0]))
        gender = "Male" if np.round(pred[0][0][0]) == 0 else "Female"
        
        self.result_label.config(text=f"Predicted Age: {age}, Predicted Gender: {gender}")

if __name__ == "__main__":
    root = tk.Tk()
    app = AgeGenderApp(root)
    root.mainloop()
