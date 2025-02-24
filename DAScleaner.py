import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import re 
import csv
import glob
from collections import deque

class DAS_cleaner:
    def __init__(self, root):
        self.root = root
        self.root.title("DAS Cleaner")
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
        self.root.bind("<e>", self.add_earthquake)
        self.root.bind("<b>", self.add_bad)
        self.root.bind("<r>", self.add_red)

        
    def load_directory(self):
        dir_path = filedialog.askdirectory()
        if not dir_path:
            return
        
        self.file_paths = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".npy")])
        
        if not self.file_paths:
            print("No .npy files found in directory.")
            return
        
        #flag list generation
        self.whale_list = [' '] * len(self.file_paths)
        self.ship_list = [' '] * len(self.file_paths)
        self.earthquake_list = [' '] * len(self.file_paths)
        self.bad_list = [' '] * len(self.file_paths)
        self.red_list = [' '] * len(self.file_paths)

        #load in the meta data
        tmpdir = dir_path
        for i in range(4):
            clist = glob.glob(tmpdir + '/Dim_Channel.txt')
            if clist:
                break
            else:
                tmpdir = os.path.split(tmpdir)[0]
    
        if i == 4:
            raise ValueError("could not find channel, frequency, or time, files within 4 levels, check directory")

        channels = np.loadtxt(tmpdir + '/Dim_Channel.txt')
        times =  np.loadtxt(tmpdir + '/Dim_Time.txt')
        self.ltimes = len(times)
        self.tdif = times[1] - times[0]
        self.lchannel = len(channels)
        self.cmin = np.min(channels)
        self.cmax = np.max(channels)

        self.seen = [0] * len(self.file_paths) 
        self.num_files_to_display = simpledialog.askinteger("Set Display Count", "How many images to display simultaneously?", minvalue=1, maxvalue=len(self.file_paths))
        self.file_index = 0

        #setting up data buffers
        self.files_showing = deque(maxlen = self.num_files_to_display)
        self.data_showing = deque(maxlen = self.num_files_to_display)
        self.flag_showing = deque(maxlen = self.num_files_to_display)

        self.fidx_minus_nfiles = 0

        self.direction = 'next'
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

        flag_list = list(zip(self.whale_list,self.ship_list,self.earthquake_list,self.bad_list,self.red_list))

        match self.direction:
            case 'next':
                if self.file_index < len(self.file_paths):
                    self.files_showing.append(self.file_paths[self.file_index])
                    self.data_showing.append(np.load(self.file_paths[self.file_index]))
                    self.flag_showing.append(''.join(flag_list[self.file_index]))

                combined_array = np.vstack(self.data_showing)

            case 'previous':
                if self.fidx_minus_nfiles >= 0:
                    self.files_showing.appendleft(self.file_paths[self.fidx_minus_nfiles])
                    self.data_showing.appendleft(np.load(self.file_paths[self.fidx_minus_nfiles]))
                    self.flag_showing.appendleft(''.join(flag_list[self.fidx_minus_nfiles]))

                elif self.fidx_minus_nfiles <0:
                    if len(self.files_showing)>1:
                        _ = self.files_showing.pop()
                        _ = self.data_showing.pop()
                        _ = self.flag_showing.pop()
                combined_array = np.vstack(self.data_showing)

            case 'flag':
                combined_array = np.vstack(self.data_showing) #keep the previously shown data
                self.flag_showing[-1] = ''.join(flag_list[self.file_index])


        image = self.array_to_photoimage(combined_array, highlight_region=self.data_showing[-1].shape, file_names=self.files_showing)
        img_obj = self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, anchor=tk.CENTER, image=image)
        self.current_images.append(image)
        self.canvas.config(scrollregion=self.canvas.bbox("all"))



        
    def array_to_photoimage(self, array, highlight_region=None, file_names=None):
        #norm_array = (array - np.min(array)) / (np.max(array) - np.min(array))  # Normalize to 0-1
        a_mean = np.mean(array,axis=0)
        norm_array = array - a_mean[None,:]
        a_std = np.std(norm_array,axis = 0)
        norm_array /=a_std[None,:]
        norm_array = (norm_array - np.min(norm_array)) / (np.max(norm_array) - np.min(norm_array))  # Normalize to 0-1
        colormap = plt.get_cmap('magma')
        color_mapped_array = (colormap(norm_array)[:, :, :3] * 255).astype(np.uint8)  # Apply Turbo colormap
        tmp = root.geometry()
        vals = re.split(r'\D+',tmp)[0:2]
        xs = int(vals[0])
        ys = int(vals[1])

        ytick_labs = np.arange(self.num_files_to_display+1)
        ytick_coords = ytick_labs * self.ltimes
        
        xstep = 10 #km
        xtick_labs = np.arange(start = self.cmin/1000, stop = self.cmax/1000, step = xstep)
        xtick_coords = np.arange(start = 0, stop = self.lchannel, step = xstep/(((self.cmax-self.cmin)/1000)/self.lchannel))

    
        fig, ax = plt.subplots(figsize=(xs/100, ys/100))  # Adjust figure size
        ax.imshow(color_mapped_array, aspect = 'auto')
        ax.set_yticks(ytick_coords,ytick_labs)
        ax.set_xticks(xtick_coords,xtick_labs)
        ax.axis('on')
        
        if highlight_region:
            h, w = highlight_region
            y_start = array.shape[0] - h  # Start position for highlighting
            ax.add_patch(plt.Rectangle((0, y_start), w-1, h-1, edgecolor='red', linewidth=2, fill=False))
        
        
        if file_names:
            y_pos = np.cumsum([d.shape[0] for d in self.data_showing])
            y_add = self.data_showing[0].shape[0] // 2
            for i, file in enumerate(file_names):
                ax.text(-10, y_pos[i]-y_add, os.path.splitext(os.path.basename(file))[0], va='center', ha='right', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))
                ax.text(color_mapped_array.shape[1] + 10, y_pos[i]-y_add ,self.flag_showing[i] ,va='center', ha='left', fontsize=10, color='white', bbox=dict(facecolor='black', alpha=0.5))
        
        fig.canvas.draw()
        
        image = Image.frombytes('RGBA', fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        plt.close(fig)
        
        return ImageTk.PhotoImage(image)
    
    def next_image(self, event=None):
        self.direction = 'next'
        if self.file_index < len(self.file_paths)-1:
            self.file_index += 1
            self.display_images()
        
    def previous_image(self, event=None):
        self.direction = 'previous'
        self.fidx_minus_nfiles = self.file_index - (self.num_files_to_display)
        if self.file_index > 0:
            self.file_index -= 1
            self.display_images()
    
    def add_whale(self, event = None):
        self.direction = 'flag'
        if self.whale_list[self.file_index] == 'W':
            self.whale_list[self.file_index] = ' '
        else:
            self.whale_list[self.file_index] = 'W'
        self.display_images()
    
    def add_ship(self, event = None):
        self.direction = 'flag'
        if self.ship_list[self.file_index] == 'S':
            self.ship_list[self.file_index] = ' '
        else:
            self.ship_list[self.file_index] = 'S'
        self.display_images()

    def add_earthquake(self,event = None):
        self.direction = 'flag'
        if self.earthquake_list[self.file_index] == 'E':
            self.earthquake_list[self.file_index] = ' '
        else:
            self.earthquake_list[self.file_index] = 'E'
        self.display_images()
    
    def add_bad(self, event = None):
        self.direction = 'flag'
        if self.bad_list[self.file_index] == 'B':
            self.bad_list[self.file_index] = ' '
        else:
            self.bad_list[self.file_index] = 'B'
        self.display_images()

    def add_red(self, event = None):
        self.direction = 'flag'
        if self.red_list[self.file_index] == 'R':
            self.red_list[self.file_index] = ' '
        else:
            self.red_list[self.file_index] = 'R'
        self.display_images()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = DAS_cleaner(root)
    root.mainloop()
