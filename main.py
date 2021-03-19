# main.py

import argparse
from pyfiglet import Figlet
from libido import Libido
from swiper import Swiper

def main():
	
	parser = argparse.ArgumentParser(description='Train a model to pick the right partners for you.')

	parser.add_argument("mode", type=str, choices=["manual", "train", "test", "auto"], help="TODO")
	parser.add_argument('--just-data', action='store_true')
	
	args = parser.parse_args()

	# Print a cute title
	f = Figlet(font='slant')
	print(f.renderText('The Wing Man'))


	# TODO: train mode
	if args.mode == "train":
		
		# Prepare dataset
		#assert os is dir data_sorted, else run the sorting algo


		# Load deep learning model class
		libido = Libido(train_data_dir)


		# Train model on dataset
		# libido.train...

	# TODO: test mode

	print("Logging into your account...")
	swiper = Swiper()

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