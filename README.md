# The Wing Man

The Wing Man is a automatic tinder CLI which knows "your type". The program runs on top of Tinder web.
It includes training, testing and inference logic (under development), meaning once you train the model to know which type of looks you find attractive, it swipes possible partners accordingly and automatically.

## Dependencies

* Requires Facebook account
* Requires Tinder account linked to Facebook account
* Requires Google Chrome web browser for desktop (>= 89.0)

- (Tkinter - REMOVE), pytorch, opencv

## Installation

* python -m virtualenv env
* source env/bin/activate
* pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
* pip install -r requirements.txt


Make sure you have the adequate [chromedriver](https://chromedriver.chromium.org/) for your OS


## Usage

``python main.py {manual, train, test, auto}``

- manual: swipe manually and label accordingly. set --just-data #TODO
- train: TODO
- test: TODO
- auto: TODO

See ``python main.py -h``for more info

## Contributing

Forks and PRs welcome
