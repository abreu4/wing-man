# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A library for image data treatment
#
# » convert: converts all images to .jpg
# » rename: rename all files in crescent order (1,2,...,n)
# » remove_duplicates: removes duplicates using md5 hash criteria
# » keep_only_pics_with_people: *undergoing tests*
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import cv2
import time
import subprocess
import tkinter as tk
from PIL import ImageTk, Image
from shutil import move, copy

from utilities import *
from detector import DetectorAPI


def rename(folder):

    """ Renames files in folder to exact numeric ascending order """

    assert os.path.isdir(folder), "Invalid data folder"

    # get all the files that are not folder in the data folder (images)
    imagefiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.jpg')]
    imagefiles.sort(key=filenumber)

    # rename according to index
    for i in range(len(imagefiles)):
        source = os.path.join(folder, imagefiles[i])
        extension = os.path.splitext(imagefiles[i])[1]
        destination = os.path.join(folder, str(i)+extension)
        os.rename(source, destination)

        print("Renamed "+str(imagefiles[i])+" -> "+str(i)+extension)

    return 1


def convert(folder):

    """ Converts every picture inside folder to JPEG """

    assert os.path.isdir(folder), "Invalid data folder"

    for filename in os.listdir(folder):
        if filename.endswith('.webp') or filename.endswith('.png'):

            # creating jpg
            imgpath = os.path.join(folder, filename)
            im = Image.open(imgpath).convert("RGB")
            im.save(os.path.join(folder, os.path.splitext(filename)[0] + '.jpg'), "jpeg")

            # deleting duplicate in webp
            os.remove(imgpath)

            print('Converted '+str(filename))

    return 1


def remove_duplicates(folder):

    """ Removes duplicate file entries inside folder """

    assert os.path.isdir(folder), "Invalid data folder"
    duplicates = []
    hash_keys = []

    for filename in os.listdir(folder):
        imgpath = os.path.join(folder, filename)
        if os.path.isfile(imgpath):
            hax = file_hash(imgpath)
            if hax not in hash_keys:
                hash_keys.append(hax)
            else:
                duplicates.append(imgpath)

    # Remove all the duplicates
    [os.remove(copycat) for copycat in duplicates]

    print('Removed '+str(len(duplicates))+' duplicates')
    return 1

def crop_to_squares(folder):
    
    """ Crops and OVERWRITES images in 'folder' to central square """

    piclist = [picture for picture in os.listdir(folder) if not os.path.isdir(picture)]
    # print(piclist)

    # TODO: - For this to work, 'piclist' must have assertions for image retrieval rather than any folders and files
    if len(piclist) < 1:
        print("\tNo pictures in source directory. Aborting...")
        return

    for i, image in enumerate(piclist):

        imagepath = os.path.join(folder, image)

        # fig = plt.figure(figsize=(10,10))

        """
        fig.add_subplot(211)
        plt.imshow(image)
        """
        image = Image.open(imagepath)
        cropped_image = _crop_to_square(image)

        """
        fig.add_subplot(212)
        plt.imshow(image)
        plt.show()
        """

        cropped_image.save(imagepath)

def keep_only_pics_with_people(folder):

    """ TODO not really removing anything yet, still under testing """

    print("Trimming dataset...")

    model_path = r"C:\Users\Tiago\PycharmProjects\thewingman\support\faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    # List all pictures in folder
    images = [os.path.join(folder, i) for i in os.listdir(folder)
              if os.path.isfile(os.path.join(folder, i))
              and i.endswith(".jpg")
              or i.endswith(".webp")]

    counter = 0
    found = False
    for img in images:

        img = cv2.imread(img)
        img = cv2.resize(img, (1280, 720))
        boxes, scores, classes, num = odapi.processFrame(img)

        # Visualization of the results of a detection.

        for i in range(len(boxes)):
            # Class 1 represents human
            if classes[i] == 1 and scores[i] > threshold:
                found=True
                box = boxes[i]
                cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)

        if not found:
            cv2.imshow("preview", img)
            cv2.waitKey()
        else:
            found = False

    print("Kept %d images" % counter)

def resize(src_folder, des_folder, width=640, height=800):

    """ Normalizes src folder images to width*height into des folder """
    """ Assumes 'src' folder's images are '.jpg' format """

    print('\nInitializing resizing subroutine')

    assert os.path.isdir(src_folder), "Invalid source data folder"
    if not os.path.isdir(des_folder):
        os.mkdir(des_folder)
        print("New destination directory created")
    print(os.listdir(des_folder))
    assert not os.listdir(des_folder), "Destination folder is not empty"

    image_list = os.listdir(src_folder)
    assert image_list is not None, "There's nothing to resize in that folder"

    dim = (width, height)
    print('Goal dimensions {:}'.format(dim))
    for image in image_list:
        src_img = os.path.join(src_folder, image)

        if os.path.isfile(src_img) and src_img.endswith('.jpg'):

            des_img = os.path.join(des_folder, image)
            img_array = cv2.imread(src_img, cv2.IMREAD_UNCHANGED)

            if img_array.shape[0] == dim[1] and img_array.shape[1] == dim[0]:
                copy(src_img, des_img)

            else:
                rs_img = cv2.resize(img_array, dim, interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(des_img, rs_img)
                print('Resized image '+image+' {:}'.format(img_array.shape))

        else:
            continue
    return 1


def split(datafolder, train_test_ratio):

    """
    Given a folder with subfolders for each class, copies and splits data across a 'test' and 'train' folder according to 'train_test_ratio'
    
    » In:
    datafolder
        |- class_x
        |- class_y

    » Out:
    datafolder
        |- class_x
        |- class_y
        |- train
        |- test

    » Algorithm:
    Make 'test' and 'train' dirs in root
    For class_x
    	Make folder for class_x inside of each dir
     	List all source image files
    	Shuffle said list
    	Define proportions accordingly
    	In range num_test, pop() to test/class_x
    	In range num_train, pop() to train/class_x
    
    (Assumes root has one folder for each of the classes and nothing else at the moment)
    """

    # Class folder names
    folders = [d for d in os.listdir(datafolder) if not d[0] == '.' and 'pycache' not in d and not os.path.isfile(d) ]

    # Create 'train' and 'test' directories
    train_path, test_path = __make_train_test_dir(datafolder)

    # Work with one class folder at a time
    for current_class in folders:

        # DEBUG: Current class
        print("Class: {}".format(current_class))

        # Create class folder inside every directory
        class_train_path, class_test_path = __spread_class_folder((train_path, test_path), current_class)

        # List all source image files
        image_list = os.listdir(os.path.join(datafolder, current_class))
        random.shuffle(image_list)

        # Extract relative proportions
        size = len(image_list)
        train_size = int(size * train_test_ratio)
        test_size = size - train_size

        print("\t Total: {} \n\t Train: {} \n\t Test: {}".format(size, train_size, test_size))

        # Copy to "train/current_class"
        for index in range(train_size):
            image_name = image_list.pop()
            image_source = __get_image_source_path(datafolder, current_class, image_name)
            image_destination = os.path.join(class_train_path, image_name)
            copy(image_source, image_destination)

        # Copy to "test/current_class"
        for index in range(test_size):
            image_name = image_list.pop()
            image_source = __get_image_source_path(datafolder, current_class, image_name)
            image_destination = os.path.join(class_test_path, image_name)
            copy(image_source, image_destination)

    return True

### Helper functions ###

def __get_image_source_path(datafolder, class_name, file):
	return os.path.join(os.path.join(datafolder, class_name), file)

def __get_image_destination_path(train_test_path, folder, file):
    return os.path.join(os.path.join(train_test_path, folder), file)

def __make_train_test_dir(datafolder):
    # Returns path relative to datafolder

    paths = [os.path.join(datafolder, x) for x in ['train', 'test']]

    try:
        [os.mkdir(path) for path in paths]
        return paths
    except FileExistsError:
        print("Train/Test folders exist. Double-check your shit...")
        exit()

def __spread_class_folder(paths, class_name):
    class_paths = [os.path.join(path, class_name) for path in paths]
    successful_class_paths = []

    if len(class_paths) > 0:
        for path in class_paths:
            try:
                os.mkdir(path)
                successful_class_paths.append(path)
            except FileExistsError:
                print("Class folder {} exists in {}.".format(class_name, path))
                exit()

    return successful_class_paths

def _crop_to_square(image):
    
    """ Crops a single image to square (edge = smallest side) from center """
    """ Input/Output is PIL.Image """

    width, height = image.size

    edge = min(width, height)

    left = (width - edge) / 2
    top = (height - edge) / 2
    right = (width + edge) / 2
    bottom = (height + edge) / 2

    return image.crop((left, top, right, bottom))