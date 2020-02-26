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

        annotation_title_frame = Frame(self)
        Label(annotation_title_frame, text="Where annotations should be saved?").pack(side=LEFT)

        annotation_input_filename_frame = Frame(self)
        self.annotations_filename = StringVar(value="annotations")
        self.annotations_path = "./{}.csv".format(self.annotations_filename.get())
        self.annotations = pd.DataFrame(columns=["emotion", "path"])
        Entry(annotation_input_filename_frame, textvariable=self.annotations_filename).pack(side=LEFT)

        annotation_buttons_frame = Frame(self)

        def make_annotate_func(self, label):
            return lambda: self.annotate(label)
        for index, category in enumerate(config["catslist"]):
            Button(annotation_buttons_frame, text=category, command=make_annotate_func(self, index), highlightbackground="#3E4149").pack(side=LEFT)
        Button(annotation_buttons_frame, text="Remove annotation", command=self.annotate("", True), highlightbackground="#3E4149").pack(side=LEFT)

        annotation_results_frame = Frame(self)
        self.annotation_result = Label(annotation_results_frame, text="No annotation selected yet.")
        self.annotation_result.pack(side=LEFT)


        predictions_labels_title_frame = Frame(self)
        self.predictions_label_title = Label(predictions_labels_title_frame, text="Predictions: ")
        self.predictions_label_title.pack(side=LEFT)
        predictions_labels_frame = Frame(self)
        self.predictions_labels = {
            cat: Label(predictions_labels_frame, text="") for cat in config["catslist"]
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
        try:
            directory = filedialog.askdirectory()
        except Exception:
            return

        self.directory = directory
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
        try:
            file = filedialog.askopenfilename()
        except Exception:
            return

        self.directory = ""
        self.current_img_idx = 0
        self.img_paths = [file]
        self.load_image()
        if not self.predictions.empty:
            self.load_predictions()

    def open_predictions(self):
        try:
            path = filedialog.askopenfilename()
        except Exception:
            return
        self.predictions = pd.read_csv(path)
        self.load_predictions()

    def load_predictions(self):
        if self.predictions.empty or len(self.img_paths) == 0:
            return

        img_predictions = self.predictions[self.predictions["path"] == self.img_paths[self.current_img_idx]]
        if not img_predictions.empty:
            for cat in config["catslist"]:
                self.predictions_labels[cat].config(text="{}: {}".format(cat, img_predictions[cat].values[0]))
            self.predictions_label_title.config(text="Predictions: ")
        else:
            self.predictions_label_title.config(text="No predictions found for this file.")
            for cat in config["catslist"]:
                self.predictions_labels[cat].config(text="")

    def load_image(self):
        path = self.img_paths[self.current_img_idx]
        img = PIL.Image.open(path)

        if img.mode == "1": # bitmap image
            self.img = PIL.ImageTk.BitmapImage(img, foreground="white")
        else:              # photo image
            self.img = PIL.ImageTk.PhotoImage(img)
        self.la.config(image=self.img, bg="#000000",
            width=self.img.width(), height=self.img.height())

    def annotate(self, label, delete=False):
        """
        :param label: label
        :param delete: if set to True, just delete the annotation for current image.
        :return:
        """
        if not self.annotations_filename:
            self.annotation_result.config(text="Please enter an annotation file name.")
            return

        current_img_path = self.img_paths[self.current_img_idx]
        self.annotations_path = "./{}.csv".format(self.annotations_filename.get())
        try:
            self.annotations = pd.read_csv(self.annotations_path)
        except FileNotFoundError:
            pass

        is_annotated = np.sum(self.annotations['path'].values == current_img_path) > 0

        if not delete:
            if is_annotated:
                self.annotations['emotion'].values[self.annotations['path'].values == current_img_path] = label
            else:
                self.annotations = self.annotations.append({"emotion": label, "path": current_img_path}, ignore_index=True)
        else:
            self.annotations.drop(self.annotations[self.annotations['path'].values == current_img_path].index, inplace=True)

        self.annotations.to_csv(self.annotations_path, index=False)
        self.load_annotation()
        self.next()

    def load_annotation(self):
        if self.annotations.empty or self.annotations_path != "./{}.csv".format(self.annotations_filename.get()):
            try:
                self.annotations = pd.read_csv(self.annotations_path)
                self.annotations_path = "./{}.csv".format(self.annotations_filename.get())
            except FileNotFoundError:
                self.annotation_result.config(text="Error while loading annotations: the file does not exist.")

        current_img_path = self.img_paths[self.current_img_idx]
        mask = self.annotations['path'].values == current_img_path
        is_annotated = np.sum(mask) > 0
        if is_annotated:
            annotation = self.annotations['emotion'].values[mask][0]
            category = config["catslist"][annotation]
            self.annotation_result.config(text="Annotated as {}".format(category))
        else:
            self.annotation_result.config(text="No annotation selected yet.")

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
