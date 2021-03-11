# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This is a visual app for picture ranking
#
# - Rank pictures in the selected folder
# - Memorize last ranked picture and start from there
# - 'labels.csv' is the output file responsible for saving every label for a given set
# - Could have a function to clear latest entry in .csv if defective, and start from there
# - 'self.i' is the picture counter
#
# Assumes a static folder, with a fixed number of immutable images with the same dimensions.
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import csv
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

from utilities import *


class Ranker:

    def __init__(self, root):

        # binding main frame
        self.root = root

        # basic appearence setup
        self.image_panel = tk.Label(self.root, text="Welcome to the chick ranker.\nPick a folder and start labelling", image=None)
        self.image_panel.bind("<Key>", self.key_press)
        self.image_panel.grid(row=5, column=0)
        self.image_panel.grid(row=5, column=0)

        self.close_button = tk.Button(self.root, text="Close", command=self.root.quit)
        self.close_button.grid(row=0, column=2)

        # setting up the folder to be labelled
        self.fold_path = None
        while self.fold_path is None:
            try:
                self.fold_path = tk.filedialog.askdirectory()  # ask which folder to label
            except:
                print("Invalid folder")
                # folder is empty
                # folder does not exist
                # no pictures in folder

        # TODO these assertions should not be permanent solution to folder quality diagnosis
        assert os.path.isdir(self.fold_path), "Specified folder does not exist"
        assert os.listdir(self.fold_path), "Specified folder is empty"

        # list and sort all the pictures in folder
        self.image_filenames = [f for f in os.listdir(self.fold_path) if
                                os.path.isfile(os.path.join(self.fold_path, f)) and f.endswith('.jpg')]
        self.image_filenames.sort(key=filenumber)
        self.image_filenames_length = len(self.image_filenames)

        # add a folder name label
        self.fold_name = os.path.split(self.fold_path)[1]
        self.foldername_label = tk.Label(self.root, text="Labelling folder ´´"+self.fold_name+"´´", bg="blue", fg="white")
        self.foldername_label.grid(row=0, column=0)

        # setting up a picture counting label
        self.pictures_left_label = tk.Label(self.root, text=None)
        self.pictures_left_label.grid(row=1, column=0)
            # text="Labelling picture " + str(self.i) + " out of " + str(self.image_filenames_length)

        # setting up the labels file
        self.labels_path = None
        self.labels_name = self.fold_name + "_labels.csv"  # computes the expected labels file
        self.labels_path = os.path.join(self.fold_path, self.labels_name)
        if os.path.isfile(self.labels_path):
            """
            This code 'if' code block will set the picture iterator 
            to the first unlabelled image's index.
            It assumes data pre-processing was run beforehand, including image resizing, duplicate
            removal and correct renaming for the file being handled
            """
            print("Found a valid labels file " + self.labels_name)

            with open(self.labels_path, mode='r') as labels_file:
                reader = csv.reader(labels_file, delimiter=',')
                nrows = sum(1 for row in reader)
                print('nrows = {}\nnfiles = {}'.format(nrows, self.image_filenames_length))
                self.i = nrows-1

                # if all the pictures are labelled, no use in continuing
                # thinner conditions could be added later, such as label quality
                if self.i >= self.image_filenames_length:
                    self.pictures_left_label.config(text="All pictures are labelled")
                    self.image_panel.config(text="Restart the app")
                    return
                else:
                    self.pictures_left_label.config(text="Labelling picture " + str(self.i+1) + " out of " + str(self.image_filenames_length))


            # TODO thoroughly check latest fully labelled image, set 'self.i' to next index
            # TODO can even use 'self.i' index to display amount of pictures left to label
            # TODO remove 'self.i' below once stuff above is working
            # TODO check validity of labels file - correct fields!

        else:
            print("No valid labels file found. Making " + self.labels_name)
            with open(self.labels_path, mode='a+', newline='') as labels_file:
                fieldnames = ['File name', 'Girl score', 'Misc score']
                writer = csv.DictWriter(labels_file, fieldnames=fieldnames)
                writer.writeheader()
                self.i = 0
                self.pictures_left_label.config(text="Labelling picture " + str(self.i+1) + " out of " + str(self.image_filenames_length))

        # setting up previous score displays
        self.prev_girl_score_display = tk.Label(self.root, text="Previous girl score: ")
        self.prev_girl_score_display.grid(row=1, column=2)
        self.prev_girl_score_display_value = tk.Label(self.root, text="None")
        self.prev_girl_score_display_value.grid(row=1, column=3)

        self.prev_misc_score_display = tk.Label(self.root, text="Previous misc score: ")
        self.prev_misc_score_display.grid(row=2, column=2)
        self.prev_misc_score_display_value = tk.Label(self.root, text="None")
        self.prev_misc_score_display_value.grid(row=2, column=3)

        # setting up current score displays
        self.girl_score_display = tk.Label(self.root, text="Girl score: ")
        self.girl_score_display.grid(row=3, column=2)
        self.girl_score_display_value = tk.Label(self.root, text="None")
        self.girl_score_display_value.grid(row=3, column=3)

        self.misc_score_display = tk.Label(self.root, text="Misc score: ")
        self.misc_score_display.grid(row=4, column=2)
        self.misc_score_display_value = tk.Label(self.root, text="None")
        self.misc_score_display_value.grid(row=4, column=3)

        # load first unlabelled picture to frame
        self.load_picture()
        self.image_panel.focus_set()

    def key_press(self, event):

        """
        3 possible cases
            a. clicked a valid number and nothing is ranked yet
                - add a score to the girl label
            b. clicked a valid number and girl is already ranked
                - add a score to the misc label
                - save image_filename, image_hash, gscore and mscore to .csv file
                - load next figure
            c. clicked 'R' triggering a reset on anything labelled yet
        """

        # extract the pressed key from the event
        kp = str(event.char)

        try:
            kp = float(kp)
            if kp > 5:
                print("Invalid score (1 to 5 expected)")
                return
            score = float(kp) / 5
            g_cur = self.girl_score_display_value.cget("text")
            m_cur = self.misc_score_display_value.cget("text")

            # a. clicked a valid number and nothing is ranked yet
            if g_cur == "None" and m_cur == "None":
                self.girl_score_display_value.config(text=str(score))

            # b. clicked a random number and girl is already ranked
            elif g_cur is not "None" and m_cur == "None":
                self.misc_score_display_value.config(text=str(score))
                new_row = {'File name': self.image_filenames[self.i],
                           'Girl score': float(g_cur),
                           'Misc score': score}
                self.save_entry(new_row)
                self.i += 1
                self.clear_scores()
                self.load_picture()

        except ValueError:
            print("Invalid scoring key")

        # test print for the current pressed key
        print("Pressed", kp)  # repr(event.char))

    def clear_scores(self):
        """ clears scores for current picture and loads it to scores for previous picture"""
        self.prev_girl_score_display_value.config(text=self.girl_score_display_value.cget("text"))
        self.prev_misc_score_display_value.config(text=self.misc_score_display_value.cget("text"))

        self.girl_score_display_value.config(text="None")
        self.misc_score_display_value.config(text="None")

    def load_picture(self):

        if self.i+1 > self.image_filenames_length:
            self.foldername_label.config(text="Done labelling folder ´´"+self.fold_name+"´´", bg="green", fg="black")
            self.image_panel.unbind("<Key>")
            self.image_panel.config(image="", text="")
            self.pictures_left_label.config(text="")

        else:
            self.pictures_left_label.config(text="Labelling picture " + str(self.i+1) + " out of " + str(self.image_filenames_length))
            image_path = os.path.join(self.fold_path, self.image_filenames[self.i])
            if os.path.isfile(image_path):
                image_hash = file_hash(image_path)
                # TODO check if it's in .csv - discard hash?
                resized = Image.open(image_path).resize((320, 400), Image.ANTIALIAS)
                image_frame = ImageTk.PhotoImage(resized)
                self.image_panel.config(image=image_frame)
                self.image_panel.image = image_frame

    def save_entry(self, new_row):

        """ given image and new row, adds new line to .csv file """

        with open(self.labels_path, mode='a', newline='') as labels_file:
            writer = csv.DictWriter(labels_file, new_row.keys())
            writer.writerow(new_row)


def run():

    # load the frame and wait for the user to pick a folder to label
    r = tk.Tk()
    r.title('FMK')
    rh = Ranker(r)
    r.mainloop()

    return
