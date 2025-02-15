import os
import shutil
from tkinter import Tk, Label, Button, filedialog
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Viewer")
        self.root.geometry("800x600")

        self.image_label = Label(root)
        self.image_label.pack()

        self.prev_button = Button(root, text="Previous", command=self.show_prev_image)
        self.prev_button.pack(side="left")

        self.next_button = Button(root, text="Next", command=self.show_next_image)
        self.next_button.pack(side="right")

        self.copy_button = Button(root, text="Copy", command=self.copy_image)
        self.copy_button.pack(side="bottom")

        self.image_folder = filedialog.askdirectory(title="Select Image Folder")
        self.images = [f for f in os.listdir(self.image_folder) if f.endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
        self.current_image_index = 0

        self.show_image()

    def show_image(self):
        image_path = os.path.join(self.image_folder, self.images[self.current_image_index])
        image = Image.open(image_path)
        image = image.resize((800, 600), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def show_prev_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(self.images)
        self.show_image()

    def show_next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(self.images)
        self.show_image()

    def copy_image(self):
        #dest_folder = filedialog.askdirectory(title="Select Destination Folder")

        src_path = os.path.join(self.image_folder, self.images[self.current_image_index])
        shutil.copy(src_path, "C:/repos/yolo_experiment/training_frames")

if __name__ == "__main__":
    root = Tk()
    app = ImageViewer(root)
    root.mainloop()
