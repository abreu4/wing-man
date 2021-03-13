# main.py

import argparse
from pyfiglet import Figlet
from libido import Libido
from swiper import Swiper

CURRENT_PREDICTION_MODEL = 'trained_models/5fbd352c-adac-11ea-89e4-cc2f71f824a0.pth'
DATA = 'data/raw'
DATA_TEST = 'data/test'
DATA_TRAIN = 'data/train'

# Temporary test folders
_DATA_TRAIN = 'data/sorted'
_DATA_TEST = 'data/sorted/test'
_DATA = ''

def main():
	
	parser = argparse.ArgumentParser(description='Train a model to pick the right partners for you.')

	parser.add_argument("mode", type=str, choices=["manual", "train", "test", "auto"], help="TODO")
	parser.add_argument('--just-data', action='store_true')
	
	args = parser.parse_args()

	f = Figlet(font='slant')
	print(f.renderText('The Wing Man'))


	# TODO: train mode
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