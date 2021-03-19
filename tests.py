from utilities import *
import data
import Libido

TOY_FOLDER = "./data_test/"
REAL_FOLDER ="./data/"

""" Data processing module tests """

def test_rename(folder=TOY_FOLDER): 
	data.rename(folder)
	print("Passed rename() test")

def test_convert(folder=TOY_FOLDER):
	data.convert(folder)
	print("Passed convert() test")

def test_remove_duplicates(folder=TOY_FOLDER):
	data.remove_duplicates(folder)
	print("Passed remove_duplicates() test")

def test_crop_to_squares(folder=TOY_FOLDER):
	data.crop_to_squares(folder)
	print("Passed crop_to_squares() test")

def test_split_dataset(folder=TOY_FOLDER):
	data.split_dataset(folder, 0.8)
	print("Passed split_dataset() test")

# data processing pipeline
def test_setup_entire_dataset(folder=TOY_FOLDER):
	data.setup_entire_dataset(folder, 0.8)
	print("Successfully prepared entire dataset")

# full data setup pipeline
def test_new_training_dataset(folder=REAL_FOLDER):
	data.new_training_dataset(folder)

""" Deep learning model tests """
#def test

if __name__ == '__main__':
	"""
	test_rename()
	test_convert()
	test_remove_duplicates()
	test_crop_to_squares()
	test_split_dataset()
	test_new_training_dataset()
	"""
