# The Wing Man

The Wing Man is an automatic tinder CLI which "knows your type". 
The program runs on top of Tinder web.
Once you train a model on your taste, it swipes possible partners accordingly and automatically.

## Dependencies

* Requires Facebook account
* Requires Tinder account linked to Facebook account
* Requires Google Chrome web browser for desktop (>= 89.0)

## Installation

1. Setup virtual environment
	* `python -m virtualenv env`
	* `source env/bin/activate`
	* `pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html`
	* `pip install -r requirements.txt`

2. Create Facebook `credentials.py` file
``` 
username = "your@email.com"
password = "yourpassword" 
```

Make sure you have the adequate [chromedriver](https://chromedriver.chromium.org/) for your OS


## Usage
The wing man has 4 usage options: `python main.py {manual, train, test, auto}`
The app is meant to be used sequentially: Use `manual` to build your dataset, then `train` and `test` a predictive model, then `auto` swipe Tinder.

1. Manual: Build your dataset while swiping Tinder
	* Press 1 to swipe left, 2 to swipe right, any other key to dismiss current profile
	* Use `--just-data` flag to swipe left even when you swipe right (allows saving data without spending likes)
2. Train: Create clean dataset from extracted data and train a new predictive model
	* Use `--same-dataset` to use same data as last trained model
3. Test: Load latest model and predict 8 random test pictures
4. Auto: Load latest model and swipe tinder automatically

See ``python main.py -h``for more info

## Heads up
A few known limitations, to be addressed in later versions:
	* Tinder popups will cause unexpected behaviour (ex.: "Add Tinder to your home..."). Just clicking them with the mouse will temporarily fix the issue
	* Manually delete "__temp__" folder after finishing a manual run of the program

## Summary
This was my first full fledged deep learning project, originally developed around 2019.
I used this as a launchpad to learn more about Python, web scraping and deep learning.
If you come across any bugs or improvements, please feel free to submit a PR.

