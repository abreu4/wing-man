import os
import re
import getch
import string
import random
import torch
import urllib
from hashlib import md5
from PIL import Image

keys = {49: "left", 50: "right"}


def filenumber(element):

    """ Given relative filepath, returns integer in file's name """

    return int(os.path.splitext(element)[0])


def file_hash(filepath):

    """ Given filepath, generates unique md5 hash """

    with open(filepath, 'rb') as f:
        return md5(f.read()).hexdigest()


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def wait_4_key():
    return ord(getch.getch())

def random_string(stringLength=25):

    """ Generates a random string of fixed length """

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def save_picture_from_url_block(destination, raw_link):

    """ Save a picture from a link contained in its style block """
    """ Returns (download_success, save_path) -> (bool, str) """

    link = re.search("(?P<url>https?://[^\s'\"]+)", raw_link).group("url")  # extracting just the url string from said block
    path = urllib.parse.urlparse(link).path # get the link from the url element
    name = random_string()+os.path.splitext(path)[1] # define a name for the picture

    return urllib.request.urlretrieve(link, os.path.join(destination, name))


def get_pillow_image(loader, image_name):

    """ Receives images and transformation and converts it to an inference-ready variable """

    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)

    return image

### XPATH helpers ###
def get_tinder_login_button_xpath():
    return "//*[@class='button Lts($ls-s) Z(0) CenterAlign Mx(a) Cur(p) Tt(u) Ell Px(24px) Px(20px)--s Py(0) Bdrs(0) Mih(40px) Fw($semibold) focus-button-style Fz($s) Bdrs(4px) Bg(#fff) C($c-pink) Bg($primary-gradient):h C(#fff):h']" 

def get_tinder_login_with_facebook_xpath():
    return "//*[@aria-label='Log in with Facebook']"

def get_current_profile_xpath():
    return '//div[@class="Toa(n) Wc($transform) Prs(1000px) Bfv(h) Ov(h) W(100%) StretchedBox Bgc($c-placeholder) Bdrs(8px)"][@aria-hidden="false"]'

def get_image_blocks_xpath():
    return './/span[@class="keen-slider__slide Wc($transform) Fxg(1)"]'

def get_image_container_xpath():
    return './div[@class="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox"]'