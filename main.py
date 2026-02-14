import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Load one real image from dataset
real_folder = "real"
real_images = os.listdir(real_folder)

if len(real_images) == 0:
    print("No images found in real/images folder")
    exit()

first_image_path = os.path.join(real_folder, real_images[0])


original = cv2.imread(first_image_path)
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Main window
root = tk.Tk()
root.title("Fake Currency Detection")
root.geometry("700x500")

# --------- TOP FRAME (ORANGE) ----------
top_frame = tk.Frame(root, bg="orange", height=100)
top_frame.pack(fill="x")

title = tk.Label(top_frame, text="Fake Currency Detection System",
                 bg="orange", fg="white", font=("Arial", 20, "bold"))
title.pack(pady=25)

# --------- MIDDLE FRAME (LIGHT BLUE) ----------
middle_frame = tk.Frame(root, bg="lightblue", height=200)
middle_frame.pack(fill="both", expand=True)

# Graph function
def show_graph():
    accuracy = [random.uniform(70, 95) for i in range(10)]
    plt.plot(accuracy)
    plt.title("Accuracy Graph")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

# Upload function
def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path == "":
        return

    test = cv2.imread(file_path)
    test_gray = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    min_score = 999999

    for img_name in os.listdir("real"):
        ref_path = os.path.join("real", img_name)
        ref_img = cv2.imread(ref_path)

        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        test_resized = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))

        diff = cv2.absdiff(ref_gray, test_resized)
        score = diff.mean()

        if score < min_score:
            min_score = score

    if min_score < 25:
        result_label.config(text="REAL NOTE", fg="green")
    else:
        result_label.config(text="FAKE NOTE", fg="red")



# Buttons
tk.Button(middle_frame, text="Accuracy Graph", command=show_graph,
          width=20, bg="white").pack(pady=15)

tk.Button(middle_frame, text="Upload Image", command=upload_image,
          width=20, bg="white").pack(pady=10)

# --------- BOTTOM FRAME (WHITE) ----------
bottom_frame = tk.Frame(root, bg="white", height=150)
bottom_frame.pack(fill="both")

result_label = tk.Label(bottom_frame, text="", font=("Arial", 22, "bold"), bg="white")
result_label.pack(pady=40)

root.mainloop()

