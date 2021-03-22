# main.py

import argparse
import os
from utilities import folder_assertions
from pyfiglet import Figlet
from libido import Libido
from swiper import Swiper
from data import new_training_dataset

DATA_DIR = "data/" # Should only contain left/ and right/ class folders
SORTED_DATA_DIR = "data_sorted/"
MODEL_DIR = "trained_models/"
TMP_DATA_DIR = "__temp__/"

def main():
	
	parser = argparse.ArgumentParser(description='Train a model to pick the right partners for you.')

	parser.add_argument("mode", type=str, choices=["manual", "train", "test", "auto"], help="See README.md for more info on each mode")
	parser.add_argument('--just-data', action='store_true')
	parser.add_argument('--same-dataset', action='store_true', help="Uses same dataset built for last training run")
	parser.add_argument('--with-model', type=str, help="Specify a trained model. Uses latest by default")
	
	args = parser.parse_args()

	# Print a cute title
	f = Figlet(font='slant')
	print(f.renderText('The Wing Man'))

	# Assign directory variables
	folder_assertions([DATA_DIR, SORTED_DATA_DIR, TMP_DATA_DIR, TMP_DATA_DIR, MODEL_DIR])
	folder_assertions([os.path.join(SORTED_DATA_DIR, "train/"), os.path.join(SORTED_DATA_DIR, "test/")])
	folder_assertions([os.path.join(DATA_DIR, "left/"), os.path.join(DATA_DIR, "right/")])

	sorted_data_dir = SORTED_DATA_DIR

	# Training mode
	if args.mode == "train":
		
		# Create and preprocess new dataset from unsorted data directory
		if args.same_dataset:
			assert os.path.isdir(dataset), "No training folder"
		else:
			print("Preparing new dataset from existing data")
			sorted_data_dir = new_training_dataset(DATA_DIR, SORTED_DATA_DIR)
			

		# Initialize CNN model with new dataset
		libido = Libido(sorted_data_dir=sorted_data_dir, trained_models_dir=MODEL_DIR, temp_data_dir=TMP_DATA_DIR)

		# Train model on dataset
		libido.train_model(num_epochs=5)

		return

	# Testing mode
	if args.mode == "test":

		libido = Libido(sorted_data_dir=sorted_data_dir, trained_models_dir=MODEL_DIR, temp_data_dir=TMP_DATA_DIR)
		
		if args.with_model: libido.show_pretrained_model(args.with_model)
		else: libido.show_pretrained_model()

		return True

	print("Logging into your account...")
	swiper = Swiper(data_dir=DATA_DIR, temp_data_dir=TMP_DATA_DIR)

	# Log into facebook
	if swiper.fb_login():

		# If valid, log into Tinder web
		if swiper.tinder_login:

			print("Press 1 to swipe left, 2 to swipe right, any other key to ignore current profile")

			if (args.mode == "auto"):
				libido = Libido(sorted_data_dir=sorted_data_dir, trained_models_dir=MODEL_DIR, temp_data_dir=TMP_DATA_DIR)
				swiper.swipe(just_data_extraction=args.just_data, auto=True, lib=libido)
			else:
				swiper.swipe(just_data_extraction=args.just_data, auto=False)

		else:
			print('Tinder login failed')

	else:
		print('Facebook login failed')

	exit()

if __name__ == '__main__':
    main()