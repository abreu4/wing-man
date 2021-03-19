# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# A library for image data preprocessing
#
# » rename: rename all files in crescent order (1,2,...,n)
# » convert: converts all images to .jpg
# » remove_duplicates: removes duplicates (md5 hash)
# » crop_to_squares: crop pictures to squares
# » keep_only_pics_with_people: (TODO) keep only images with people
# » resize: resize and duplicate files to new folder using custom dimensions
# » split_dataset: create train and test folders with given ratio of instances
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import os
import cv2
import time
import subprocess
from PIL import ImageTk, Image
from shutil import move, copy, rmtree, copytree
from utilities import *


def rename(folder):

    """ Renames files in folder to exact numeric ascending order (inplace) """
    """ TODO: Ensure duplicates get handled correctly """

    assert os.path.isdir(folder), "Invalid data folder"
    imagefiles = __get_image_names_in(folder)
    assert len(imagefiles) > 0, "No pictures in folder"

    counter = 0
    # Rename according to index
    for i, img in enumerate(imagefiles):
        source = os.path.join(folder, img)
        extension = os.path.splitext(img)[1]
        destination = os.path.join(folder, str(i)+extension)
        
        os.rename(source, destination)
        counter ++ 

    print("Renamed "+str(counter)+" files in "+folder)

    return True


def convert(folder):

    """ Converts every picture inside folder to JPEG (inplace) """

    assert os.path.isdir(folder), "Invalid data folder"

    for filename in os.listdir(folder):
        if filename.endswith('.webp') or filename.endswith('.png'):

            # Create jpg
            imgpath = os.path.join(folder, filename)
            im = Image.open(imgpath).convert("RGB")
            im.save(os.path.join(folder, os.path.splitext(filename)[0] + '.jpg'), "jpeg")

            # Delete webp duplicates
            os.remove(imgpath)

            print('Converted '+str(filename))

    return True


def remove_duplicates(folder):

    """ Removes duplicate file entries inside folder """

    assert os.path.isdir(folder), "Invalid data folder"
    imagefiles = __get_image_names_in(folder)
    assert len(imagefiles) > 1, "No duplicates in folder"

    duplicates = []
    hash_keys = []

    for image in imagefiles:

        source = os.path.join(folder, image)
        hax = file_hash(source)

        if hax not in hash_keys: hash_keys.append(hax)
        else: duplicates.append(source)

    # Remove all the duplicates
    [os.remove(copycat) for copycat in duplicates]

    print('Removed '+str(len(duplicates))+' duplicates')
    return True

def crop_to_squares(folder):
    
    """ Crops images in 'folder' to central square (inplace) """

    assert os.path.isdir(folder), "Invalid data folder"
    imagefiles = __get_image_names_in(folder)
    assert len(imagefiles) > 0, "No pictures in folder"

    for i, image in enumerate(imagefiles):

        imagepath = os.path.join(folder, image)
        image = Image.open(imagepath)
        
        cropped_image = _crop_to_square(image)
        cropped_image.save(imagepath)

    return True

def keep_only_pics_with_people(folder):

    """ TODO not really removing anything yet, still under testing """
    """
    print("Trimming dataset...")

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
    """

def resize(src_folder, des_folder, width=640, height=800):

    """ Normalizes src folder images to width*height into des folder """
    """ Assumes src folder's images are '.jpg' format """

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


def split_dataset(folder, train_test_ratio, class_names=["left", "right"]):

    """
    Given a folder with subfolders for each class, copies and splits data across a 'test' and 'train' folder according to 'train_test_ratio' (0,1)
    Works for any 'class_names'; Default is left/right for current project.

    » In:
    folder
        |- class_x
        |- class_y
        |- ...

    » Out:
    folder
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
    folders = [d for d in os.listdir(folder) if d in class_names]
    assert len(folders) == len(class_names), "Couldn't find specified class folders"

    # Create 'train' and 'test' directories
    train_path, test_path = __make_train_test_dir(folder)

    # Work with one class folder at a time
    for current_class in folders:

        # DEBUG: Current class
        print("Class: {}".format(current_class))

        # Create class folder inside every directory
        class_train_path, class_test_path = __spread_class_folder((train_path, test_path), current_class)

        # List all source image files
        image_list = os.listdir(os.path.join(folder, current_class))
        random.shuffle(image_list)

        # Extract relative proportions
        size = len(image_list)
        train_size = int(size * train_test_ratio)
        test_size = size - train_size

        print("\t Total: {} \n\t Train: {} \n\t Test: {}".format(size, train_size, test_size))

        # Copy to "train/current_class"
        for index in range(train_size):
            image_name = image_list.pop()
            image_source = __get_image_source_path(folder, current_class, image_name)
            image_destination = os.path.join(class_train_path, image_name)
            copy(image_source, image_destination)

        # Copy to "test/current_class"
        for index in range(test_size):
            image_name = image_list.pop()
            image_source = __get_image_source_path(folder, current_class, image_name)
            image_destination = os.path.join(class_test_path, image_name)
            copy(image_source, image_destination)

    return True

### Data preprocessing pipeline ###

def setup_entire_dataset(folder, train_test_ratio=0.8):

    """ 
    Given a folder with class subfolders
        Preprocess data in each class subfolder
        Copy and populate train and test folders
    Delete original class subfolders

    WARNING: Alters the data folder inplace!
    """

    class_folders = os.listdir(folder)

    # Preprocess data in each class folder
    for f in class_folders:

        subfolder = os.path.join(folder, f)
        
        assert rename(subfolder) == True, "Failed while renaming files"
        assert convert(subfolder) == True, "Failed while converting files to jpg"
        assert remove_duplicates(subfolder) == True, "Failed while removing duplicates"
        assert crop_to_squares(subfolder) == True, "Failed while cropping pictures to squares"
        # assert keep_only_pics_with_people(folder) == True, "Failed while removing pictures with no people"

    # Split class folders into train/test folders
    assert split_dataset(folder, train_test_ratio) == True, "Failed while splitting dataset into train/test folders"

    # Remove original class folders
    for f in class_folders:
        rmtree(os.path.join(folder, f))

    return True

def new_training_dataset(folder, train_test_ratio=0.8):

    sorted_folder = "./data_sorted/"
    
    breakpoint()
    # Remove old sorted folder
    if os.path.isdir(sorted_folder): rmtree(sorted_folder) 

    # Create sorted folder and copy original data to it
    copytree(folder, sorted_folder)

    # Preprocess data
    setup_entire_dataset(sorted_folder)

    return True
        

# Preprocess left and right folders ()


### Helper functions ###

def __get_image_names_in(folder):
    return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and (f.endswith('.jpg') or f.endswith('.webp') or f.endswith('.png'))]


def __get_image_source_path(folder, class_name, file):
	return os.path.join(os.path.join(folder, class_name), file)

def __get_image_destination_path(train_test_path, folder, file):
    return os.path.join(os.path.join(train_test_path, folder), file)

def __make_train_test_dir(folder):
    # Returns path relative to folder

    paths = [os.path.join(folder, x) for x in ['train', 'test']]

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