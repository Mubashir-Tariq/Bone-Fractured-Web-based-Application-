import customtkinter as ctk
from tkinter import filedialog
from tkinter import Label, Frame
from ultralytics import YOLO
from PIL import Image, ImageTk

# Load the YOLOv8 model
model = YOLO('best.pt')

# Global variable to store image path
img_path = None

# Function to upload image
def upload_image():
    global img_path
    img_path = filedialog.askopenfilename()

    if img_path:
        img = Image.open(img_path)
        img = img.resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)

        img_label.config(image=img_tk)
        img_label.image = img_tk

# Function to detect disease
def detect_disease():
    global img_path

    if not img_path:
        print("Please upload an image first!")
        return

    results = model(img_path)

    if results and len(results) > 0:
        print("Detections found!")
        result = results[0]
        result_img = result.plot()
        result_img = Image.fromarray(result_img)
        result_img = result_img.resize((400, 400))
        result_img_tk = ImageTk.PhotoImage(result_img)

        img_label.config(image=result_img_tk)
        img_label.image = result_img_tk
    else:
        print("No detections found.")

# Set up the main window
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

window = ctk.CTk()
window.title("Bone Fracture Detection System")
window.geometry("1000x800")
window.configure(bg="#f0f0f0")

# Configure grid layout to center elements
window.grid_rowconfigure(0, weight=1)
window.grid_rowconfigure(1, weight=1)
window.grid_rowconfigure(2, weight=1)
window.grid_columnconfigure(0, weight=1)

# Header frame
header_frame = Frame(window, bg="#800000")
header_frame.grid(row=0, column=0, sticky="nsew", pady=10)

# Logo image
logo = Image.open("1.png")
logo = logo.resize((100, 100))
logo_tk = ImageTk.PhotoImage(logo)

# Logo Label
logo_label = Label(header_frame, image=logo_tk, bg="#800000")
logo_label.pack(side="left", padx=20, pady=10)

# Header Text
header_text = Label(header_frame, text="Bone Fracture Detection", font=("Serif", 30, "bold"), bg="#800000", fg="white", pady=10)
header_text.pack(side="left", padx=10)

# Image display frame
img_frame = Frame(window, bd=2, relief="groove", bg="white", width=400, height=400)
img_frame.grid(row=1, column=0, pady=20, sticky="n")
img_frame.pack_propagate(False)

# Image Label
img_label = Label(img_frame, bg="white", width=400, height=400)
img_label.pack()

# Button frame
button_frame = ctk.CTkFrame(window, fg_color="#800000")
button_frame.grid(row=2, column=0, pady=30, sticky="n")

# Upload Button
upload_button = ctk.CTkButton(button_frame, text="Upload Image", command=upload_image, font=("Helvetica", 16, "bold"), fg_color="#ffcccc", text_color="black", corner_radius=10)
upload_button.grid(row=0, column=0, padx=20, pady=10)

# Detect Button
detect_button = ctk.CTkButton(button_frame, text="Detect Fracrured", command=detect_disease, font=("Helvetica", 16, "bold"), fg_color="#ccffcc", text_color="black", corner_radius=10)
detect_button.grid(row=0, column=1, padx=20, pady=10)

# Run the application
window.mainloop()
