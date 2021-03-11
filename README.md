# The Wing Man

The Wing Man is a automatic tinder CLI which knows "your type". The program runs on top of Tinder web.
It includes training, testing and inference logic (under development), meaning once you train the model to know which type of looks you find attractive, it swipes possible partners accordingly and automatically.

## Installation

All required packages are installed through anaconda

* ``conda create --name <env_name>``
* ``conda activate <env_name>``


Make sure you have the adequate chromedriver for your OS

## Usage

``python main.py {train, test, infer, data}``

- data: create your own dataset from tinder profiles
- train: train a model on your dataset, saves model to trained_models/
- test: preview model of your choice on test dataset
- infer: automatically evaluate all pictures from a profile and swipe left or right accordingly

See ``python main.py -h``for more info

## Contributing

Forks and PRs welcome
