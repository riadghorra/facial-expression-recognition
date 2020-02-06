#!/usr/bin/env python
##############################################################################
# Copyright (c) 2012 Hajime Nakagami<nakagami@gmail.com>
# All rights reserved.
# Licensed under the New BSD License
# (http://www.freebsd.org/copyright/freebsd-license.html)
##############################################################################

import PIL.Image
import os
import pandas as pd
import json

try:
    from Tkinter import *
    import tkFileDialog as filedialog
except ImportError:
    from tkinter import *
    from tkinter import filedialog
import PIL.ImageTk

with open('config.json') as json_file:
    config = json.load(json_file)

class App(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.directory = ""
        self.img_paths = []
        self.current_img_idx = 0
        self.img = None
        self.predictions = pd.DataFrame()

        self.master.title('Image Viewer')

        buttons_frame = Frame(self)
        Button(buttons_frame, text="Open predictions", command=self.open_predictions, highlightbackground="#3E4149").pack(side=LEFT)
        Button(buttons_frame, text="Open Directory", command=self.open_dir, highlightbackground="#3E4149").pack(side=LEFT)
        Button(buttons_frame, text="Open File", command=self.open_file, highlightbackground="#3E4149").pack(side=LEFT)
        Button(buttons_frame, text="Prev", command=self.prev, highlightbackground="#3E4149").pack(side=LEFT)
        Button(buttons_frame, text="Next", command=self.next, highlightbackground="#3E4149").pack(side=LEFT)

        predictions_labels_title_frame = Frame(self)
        self.predictions_label_title = Label(predictions_labels_title_frame, text="Predictions: ")
        self.predictions_label_title.pack(side=LEFT)
        predictions_labels_frame = Frame(self)
        self.predictions_labels = {
            cat: Label(predictions_labels_frame, text="{}: None".format(cat)) for cat in config["catslist"]
        }
        for label in self.predictions_labels.values():
            label.pack(side=LEFT)

        buttons_frame.pack(side=TOP, fill=BOTH)
        predictions_labels_title_frame.pack(side=TOP, fill=BOTH)
        predictions_labels_frame.pack(side=TOP, fill=BOTH)

        self.la = Label(self)
        self.la.pack()

        self.pack()

    def open_dir(self):
        self.directory = filedialog.askdirectory()
        self.current_img_idx = 0
        self.img_paths = []

        for r, d, f in os.walk(self.directory):
            for file in f:
                if any(extension in file for extension in ['.jpeg', '.jpg', '.png']):
                    self.img_paths.append(os.path.join(r, file))

        if len(self.img_paths):
            self.load_image()

        if not self.predictions.empty:
            self.load_predictions()

    def open_file(self):
        self.directory = ""
        self.current_img_idx = 0
        self.img_paths = [filedialog.askopenfilename()]
        self.load_image()
        if not self.predictions.empty:
            self.load_predictions()

    def open_predictions(self):
        path = filedialog.askopenfilename()
        self.predictions = pd.read_csv(path)
        self.load_predictions()

    def load_predictions(self):
        if self.predictions.empty or len(self.img_paths) == 0:
            return

        img_predictions = self.predictions[self.predictions["path"] == self.img_paths[self.current_img_idx]]
        for cat in config["catslist"]:
            self.predictions_labels[cat].config(text="{}: {}".format(cat, img_predictions[cat].values[0]))

    def load_image(self):
        path = self.img_paths[self.current_img_idx]
        img = PIL.Image.open(path)

        if img.mode == "1": # bitmap image
            self.img = PIL.ImageTk.BitmapImage(img, foreground="white")
        else:              # photo image
            self.img = PIL.ImageTk.PhotoImage(img)
        self.la.config(image=self.img, bg="#000000",
            width=self.img.width(), height=self.img.height())


    def prev(self):
        self.current_img_idx = (self.current_img_idx - 1) % len(self.img_paths)
        self.load_image()
        self.load_predictions()

    def next(self):
        self.current_img_idx = (self.current_img_idx + 1) % len(self.img_paths)
        self.load_image()
        self.load_predictions()


if __name__ == "__main__":
    app = App(); app.mainloop()


# TODO save predictions as csv (img_path, prediction)
# TODO button load predictions