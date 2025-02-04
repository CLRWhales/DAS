import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import re 
import csv

class NumpyViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NumPy Array Viewer")
        self.root.geometry("800x600")  # Make the window larger
        self.root.resizable(True, True)  # Allow resizing
        
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.load_button = tk.Button(self.button_frame, text="Select Directory", command=self.load_directory)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.second_button = tk.Button(self.button_frame, text="Save Flags", command=self.savetable)
        self.second_button.pack(side=tk.LEFT, padx=5, pady=10)
        self.load_button.pack(pady=10)

        self.file_index = 0
        self.num_files_to_display = 3  # Default number of images to display
        self.file_paths = []
        self.current_images = []

        
        #keybindings
        self.root.bind("<Down>", self.next_image)
        self.root.bind("<Up>", self.previous_image)
        self.root.bind("<s>", self.add_ship)
        self.root.bind("<w>", self.add_whale)
        self.root.bind("<e>",self.add_earthquake)
        
    def load_directory(self):
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        
        self.file_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")])
        
        if not self.file_paths:
            print("No .npy files found in directory.")
            return
        self.whale_list = [''] * len(self.file_paths)
        self.ship_list = [''] * len(self.file_paths)
        self.earthquake_list = [''] * len(self.file_paths)
        self.seen = [0] * len(self.file_paths) 
        self.num_files_to_display = simpledialog.askinteger("Set Display Count", "How many images to display simultaneously?", minvalue=1, maxvalue=len(self.file_paths))
        self.file_index = 0
        self.display_images()
    
    def savetable(self):
        dnames = [os.path.basename(file) for file in self.file_paths]
        rows = zip(dnames,self.whale_list,self.ship_list,self.earthquake_list,self.seen)
        fname = os.path.split(self.file_paths[1])[0] + '/id_flag.csv'
        with open(fname, 'w',encoding="ISO-8859-1") as f:
            writer = csv.writer(f)
            writer.writerow(['file_name','whale_flag','ship_flag','earthquake_flag','seen_flag'])
            for row in rows:
                writer.writerow(row)
        
    def display_images(self):
        self.canvas.delete("all")
        self.current_images = []
        
        self.end_index = self.file_index + self.num_files_to_display
        files_to_show = self.file_paths[self.file_index:self.end_index]
        
        arrays = [np.load(file) for file in files_to_show]
        combined_array = np.vstack(arrays)  # Stack images vertically
        
        image = self.array_to_photoimage(combined_array, highlight_region=arrays[-1].shape, file_names=files_to_show)
        img_obj = self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor=tk.CENTER, image=image)
        self.current_images.append(image)
        
        self.canvas.config(scrollregion=self.canvas.bbox("all"))
        
    def array_to_photoimage(self, array, highlight_region=None, file_names=None):
        norm_array = (array - np.min(array)) / (np.max(array) - np.min(array))  # Normalize to 0-1
        colormap = plt.get_cmap('turbo')
        color_mapped_array = (colormap(norm_array)[:, :, :3] * 255).astype(np.uint8)  # Apply Turbo colormap
        tmp = root.geometry()
        vals = re.split(r'\D+',tmp)[0:2]
        xs = int(vals[0])
        ys = int(vals[1])
        fig, ax = plt.subplots(figsize=(xs/100, ys/100))  # Adjust figure size
        ax.imshow(color_mapped_array, aspect = 'auto')
        ax.axis('off')
        
        if highlight_region:
            h, w = highlight_region
            y_start = array.shape[0] - h  # Start position for highlighting
            ax.add_patch(plt.Rectangle((0, y_start), w-1, h-1, edgecolor='red', linewidth=2, fill=False))
        
        if file_names:
            for i, file in enumerate(file_names):
                y_pos = sum(np.load(f).shape[0] for f in file_names[:i]) + np.load(file).shape[0] // 2
                ax.text(-10, y_pos, os.path.basename(file), va='center', ha='right', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))
                flagtext = self.whale_list[self.file_index:self.end_index][i]+ ' ' + self.ship_list[self.file_index:self.end_index][i] + ' ' + self.earthquake_list[self.file_index:self.end_index][i]
                ax.text(color_mapped_array.shape[1] + 10, y_pos,flagtext ,va='center', ha='left', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))
        fig.canvas.draw()
        
        image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close(fig)
        
        return ImageTk.PhotoImage(image)
    
    def next_image(self, event=None):
        lowestidx = self.file_index + self.num_files_to_display-1
        if self.file_index + self.num_files_to_display < len(self.file_paths):
            self.file_index += 1
            self.display_images()
            self.seen[lowestidx] = 1
            self.seen[lowestidx+1] = 1
        
    def previous_image(self, event=None):
        if self.file_index > 0:
            self.file_index -= 1
            self.display_images()
    
    def add_whale(self, event = None):
        #print('whale!')
        lowestidx = self.file_index + self.num_files_to_display-1
        if self.whale_list[lowestidx] == 'W':
            self.whale_list[lowestidx] = ''
        else:
            self.whale_list[lowestidx] = 'W'
        self.display_images()
    
    def add_ship(self, event = None):
        #print('ship!')
        lowestidx = self.file_index + self.num_files_to_display-1
        if self.ship_list[lowestidx] == 'S':
            self.ship_list[lowestidx] = ''
        else:
            self.ship_list[lowestidx] = 'S'
        self.display_images()

    def add_earthquake(self,event = None):
        lowestidx = self.file_index + self.num_files_to_display-1
        if self.earthquake_list[lowestidx] == 'E':
            self.earthquake_list[lowestidx] = ''
        else:
            self.earthquake_list[lowestidx] = 'E'
        self.display_images()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = NumpyViewerApp(root)
    root.mainloop()
