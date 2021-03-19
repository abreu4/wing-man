# main.py

import argparse
from pyfiglet import Figlet
from libido import Libido
from swiper import Swiper
from data import new_training_dataset

DATA_DIR = "./data/" # Should only contain left/ and right/ class folders
SORTED_DATA_DIR = "./data_sorted/"
MODEL_DIR = "./trained_models/"

def main():
	
	parser = argparse.ArgumentParser(description='Train a model to pick the right partners for you.')

	parser.add_argument("mode", type=str, choices=["manual", "train", "test", "auto"], help="See README.md for more info on each mode")
	parser.add_argument('--just-data', action='store_true')
	parser.add_argument('--rebuild-dataset', action='store_true')
	parser.add_argument('--with-model', type=str, help="Specify a trained model. Uses latest by default")
	
	args = parser.parse_args()

	# Print a cute title
	f = Figlet(font='slant')
	print(f.renderText('The Wing Man'))

	# Assign directory variables
	dataset = SORTED_DATA_DIR

	# Training mode
	if args.mode == "train":
		
		# Create and preprocess new dataset from unsorted data directory
		if args.rebuild_dataset:
			print("Preparing new dataset from existing data")
			dataset = new_training_dataset(DATA_DIR, SORTED_DATA_DIR)
		else:
			assert os.path.isdir(dataset), "Invalid training folder (Hint: ommit --rebuild-dataset flag)"

		# Initialize CNN model with new dataset
		libido = Libido(train_data_dir=dataset, trained_models_dir=MODEL_DIR)

		# Train model on dataset
		libido.train_model(num_epochs=5)

		return

	# Testing mode
	if args.mode == "test":

		libido = Libido(train_data_dir=dataset, trained_models_dir=MODEL_DIR)
		
		if args.with_model: libido.show_pretrained_model(args.with_model)
		else: libido.show_pretrained_model()

		return True

	print("Logging into your account...")
	swiper = Swiper(data_dir=DATA_DIR)

	# Log into facebook
	if swiper.fb_login():

		# If valid, log into Tinder web
		if swiper.tinder_login:

			if args.mode == "manual":

				print("Press 1 to swipe left, 2 to swipe right, any other key to ignore current profile")
				swiper.manual_swipe(just_data_extraction=args.just_data)

			elif args.mode == "auto":
				# TODO
				return

		else:
			print('Tinder login failed')

	else:
		print('Facebook login failed')

	exit()

if __name__ == '__main__':
    main()