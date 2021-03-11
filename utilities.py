import os
import string
import random
import torch
from hashlib import md5
from PIL import Image
#import msvcrt as micro


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
    return micro.getch()

def random_string(stringLength=25):

    """ Generates a random string of fixed length """

    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def get_pillow_image(loader, image_name):

    """ Receives images and transformation and converts it to an inference-ready variable """

    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)

    return image