from utilities import *
import data
from libido import Libido

DATA_DIR = "./data/" # Should only contain left/ and right/ class folders
SORTED_DATA_DIR = "./data_sorted/"
MODEL_DIR = "./trained_models/"
TMP_DATA_DIR = "./__temp__/"

""" Data processing module tests """

def test_rename(folder=SORTED_DATA_DIR): 
	data.rename(folder)
	print("Passed rename() test")

def test_convert(folder=SORTED_DATA_DIR):
	data.convert(folder)
	print("Passed convert() test")

def test_remove_duplicates(folder=SORTED_DATA_DIR):
	data.remove_duplicates(folder)
	print("Passed remove_duplicates() test")

def test_crop_to_squares(folder=SORTED_DATA_DIR):
	data.crop_to_squares(folder)
	print("Passed crop_to_squares() test")

def test_split_dataset(folder=SORTED_DATA_DIR):
	data.split_dataset(folder, 0.8)
	print("Passed split_dataset() test")

# data processing pipeline
def test_setup_entire_dataset(folder=SORTED_DATA_DIR):
	data.setup_entire_dataset(folder, 0.8)
	print("Passed setup_entire_dataset() test")

# full data setup pipeline
def test_new_training_dataset(folder=DATA_DIR):
	data.new_training_dataset(folder)
	print("Passed new_training_dataset() test")

def test_remove_pics_with_no_people(folder=SORTED_DATA_DIR):
	data.remove_pics_with_no_people(folder)
	print("Passed keep_only_pics_with_people() test")


""" Deep learning model tests """
def test_infer():
	libido = Libido(sorted_data_dir=SORTED_DATA_DIR, temp_data_dir=TMP_DATA_DIR, trained_models_dir=MODEL_DIR)
	libido.infer()
	print("Passed test_infer() test")	



if __name__ == '__main__':
	"""
	test_rename()
	test_convert()
	test_remove_duplicates()
	test_crop_to_squares()
	test_split_dataset()
	test_new_training_dataset()
	"""
	#test_test_model()
	test_infer()
	
