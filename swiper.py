import os
import re
import time
import shutil
import urllib
import tkinter
import torch
import platform
import credentials

# UNCOMMENT
#from utilities import wait_4_key, random_string

from utilities import get_pillow_image
from getpass import getpass
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

# TODO: Useful code, delete
# ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
# ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
# time.sleep(0.5)

TMP_IMAGE_DIR = "./__temp__"
DIR = './data/' # Data directory where your categories folders will be saved
LEFT = os.path.join(DIR, "left") # Left category folder
RIGHT = os.path.join(DIR, "right") # Right category folder

class Swiper():

    def __init__(self):

        option = Options()
        option.add_argument("--disable-infobars")
        option.add_argument("start-maximized")
        option.add_argument("--disable-extensions")

        # Pass the argument 1 to allow and 2 to block
        option.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 1})
        

        # Start the driver
        current_os = platform.platform()
    
        if ("Windows" in current_os):
            self.driver = webdriver.Chrome(chrome_options=option, executable_path='chromedriver.exe')
        
        elif("macOS" in current_os or "Linux" in current_os):
            self.driver = webdriver.Chrome(chrome_options=option, executable_path='./chromedriver')

    def fb_login(self):

        # Load facebook.com
        self.driver.get('https://www.facebook.com/')
        
        # Accept cookies
        self.driver.find_element_by_xpath('//*[@data-testid="cookie-policy-dialog-accept-button"]').click()

        # Enter e-mail
        a = self.driver.find_element_by_id('email')
        a.send_keys(credentials.username) # TODO: Delete for security
        
        # Enter password
        b = self.driver.find_element_by_id('pass')
        b.send_keys(credentials.password)
        
        # Click login button
        self.driver.find_element_by_name('login').click()

        # Check whether login was successful by finding the home button
        try:
            self.driver.find_element_by_xpath('//*[@aria-label="Home"]')
        except:
            return False
        return True

    @property
    def tinder_login(self):
    # TODO: Instead of waiting, should be checking for completeness

        # Load tinder.com
        self.driver.get('http://tinder.com')
        time.sleep(3)

        # Click on "Login" button
        self.driver.find_element_by_xpath(_get_tinder_login_button_xpath()).click()
        time.sleep(2)

        # Click on "Login with Facebook"
        self.driver.find_element_by_xpath(_get_tinder_login_with_facebook_xpath()).click()

        print("Ready to start swiping :)")
        return True

    def dumb_swipe(self):

        """
        Swipes right every time
        """

        actions = ActionChains(self.driver)
        print("Dumb swiping mode activated")
        time.sleep(5)
        try:
            while self.driver.find_element_by_class_name("recsCardboard"):
                actions.send_keys(Keys.ARROW_RIGHT).perform()
                time.sleep(2)
        except:
            """
            Needs extra error handling for no profiles found, popup found,
            no more likes, free super like screen, etc.
            """
            print("Something came up. Quitting...")
            self.driver.quit()


    def smart_swipe(self, model, data_transform):

        """ Takes prediction model as input """
        """ Evaluates pictures in Tinder profile """
        """ and swipes accordingly """

        self.model = model
        self.model.eval()

        with torch.no_grad():

            while True:

                # Loop until profile is found
                found_profile = False
                done = False
                while not found_profile:
                    try:
                        found_profile = self.driver.find_elements_by_class_name("react-swipeable-view-container")
                    except NoSuchElementException:
                        pass

                # Find the picture block
                image_blocks = self.driver.find_elements_by_xpath('//*[@class="recCard Ov(h) Cur(p) W(100%) Bgc($c-placeholder) StretchedBox Bdrs(8px) CenterAlign--ml Toa(n) active"]//*[@class="react-swipeable-view-container"]//*[@data-swipeable="true"]')

                # Iterates through each of the image blocks
                for i in range(len(image_blocks)):

                    # Loops until picture link is found
                    current_picture = None
                    while current_picture is None:
                        
                        try:
                            current_picture = self.driver.find_element_by_xpath('//*[@class="recCard Ov(h) Cur(p) W(100%) Bgc($c-placeholder) StretchedBox Bdrs(8px) CenterAlign--ml Toa(n) active"]//*[@class="react-swipeable-view-container"]//*[@aria-hidden="false"]//*[@class="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox"]')
                        except NoSuchElementException:
                            pass

                        # Extract picture
                        raw_link = current_picture.get_attribute('style')  # Get the full style block where link is embedded
                        link = re.search("(?P<url>https?://[^\s'\"]+)", raw_link).group("url")  # Extract url string from block
                        filename = urllib.parse.urlparse(link).path
                        new_filename = random_string()+os.path.splitext(link_path)[1]
                        full_image_path_on_device = os.path.join(TMP_IMAGE_DIR, new_filename)
                        full_image_path_on_device, _ = urllib.request.urlretrieve(link, full_image_path_on_device)  # TODO: Find out what this function is returning and add fail safes

                        # Convert it to Pillow format
                        image_to_predict = get_pillow_image(data_transform, DEBUG_IMAGE)

                        # Evaluate picture
                        # https://stackoverflow.com/questions/50063514/load-a-single-image-in-a-pretrained-pytorch-net
                        outputs = self.model(image_to_predict)
                        _ , pred = torch.max(outputs, 1)

                        print(f"result: {outputs}")

                        # TODO: store obtained left and right swipe probabilities (don't max out output)
                        # TODO: delete picture after inference

                        #np.argmax(model_ft(image_loader(data_transforms, $FILENAME)).detach().numpy())
                        #_, preds = torch.max(outputs, 1)

                    # Jump to the next picture
                    ActionChains(self.driver).send_keys(' ').perform()  # moving toward next picture

            return 1


    def data_extraction(self, just_data=False):

        """ Extracts and labels pictures from profiles """
        """ just_data flag swipes left even for profiles you labelled "right" - economizes your likes"""

        while True:

            # Loop until it finds a profile
            found_profile = False
            done = False

            while not found_profile:
                try:
                    print("here")
                    time.sleep(5)
                    found_profile = self.driver.find_elements_by_class_name(_get_current_profile_xpath())
                except NoSuchElementException:
                    pass

            print("Found a profile")

            # Find picture blocks
            image_blocks = self.driver.find_elements_by_xpath(_get_picture_cards_xpath())
            print(f"#Image blocks: {len(image_blocks)}")

            # Iterates through each of the picture blocks
            for i in range(len(image_blocks)):

                # Loops until picture link is found in current block
                current_picture = None
                while current_picture is None:
                    try:
                        current_picture = self.driver.find_element_by_xpath('//*[@class="recCard Ov(h) Cur(p) W(100%) Bgc($c-placeholder) StretchedBox Bdrs(8px) CenterAlign--ml Toa(n) active"]//*[@class="react-swipeable-view-container"]//*[@aria-hidden="false"]//*[@class="Bdrs(8px) Bgz(cv) Bgp(c) StretchedBox"]')
                    except NoSuchElementException:
                        pass

                # Extract current picture
                """ Get full style block where link is embedded """
                raw_link = current_picture.get_attribute('style')
                """ Save picture from style block """
                saved = save_picture_from(raw_link)

                # Jump to next picture
                ActionChains(self.driver).send_keys(' ').perform()  # moving toward the next picture

            # List all the pictures for the current profile
            imagefiles = [os.path.join(DIR, f) for f in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, f)) and (f.endswith('.jpg') or f.endswith('.webp'))]
            print("Saved %d pictures." % len(imagefiles))

            # TODO try/except in case user presses invalid key

            # 
            while not done:

                swiped = wait_4_key()

                # if you press 1, then pictures for current profile will be saved in 'left' folder
                if swiped == b'1':
                    [shutil.move(image, LEFT) for image in imagefiles]

                    # swipes left
                    ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                    time.sleep(0.5)
                    done = True

                # if you press 2, then pictures will go on the 'right'
                elif swiped == b'2':
                    [shutil.move(image, RIGHT) for image in imagefiles]

                    # if we're just gathering data, we save right swipes in the 'right' folder, but swipe left instead
                    if just_data:
                        # swipes left
                        ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                        time.sleep(0.5)
                    else:
                        # swipes right
                        ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
                        time.sleep(0.5)

                    done = True

                else:
                    for image in imagefiles:
                            os.remove(image)
                            print("Removed images")

                    ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                    time.sleep(0.5)

                    done = True



        return 1

def _get_tinder_login_button_xpath():
    return "//*[@class='button Lts($ls-s) Z(0) CenterAlign Mx(a) Cur(p) Tt(u) Ell Px(24px) Px(20px)--s Py(0) Bdrs(0) Mih(40px) Fw($semibold) focus-button-style Fz($s) Bdrs(4px) Bg(#fff) C($c-pink) Bg($primary-gradient):h C(#fff):h']" 

def _get_tinder_login_with_facebook_xpath():
    return "//*[@aria-label='Log in with Facebook']"

def _get_current_profile_xpath():
    #return "//*[contains(text(), '{0}')]".format("recsCardboard")
    return "//*[@class='recsCardboard__cardsContainer H(100%) Pos(r) Z(1)']"
    

def _get_picture_cards_xpath():
    return "//*[@class='keen-slider__slide Wc($transform) Fxg(1)']"

def save_picture_from(raw_link):
# Save a picture from a link contained in its style block
    
    link = re.search("(?P<url>https?://[^\s'\"]+)", raw_link).group("url")  # extracting just the url string from said block
    path = urllib.parse.urlparse(link).path # get the link from the url element
    name = random_string()+os.path.splitext(path)[1] # define a name for the picture
    
    return urllib.request.urlretrieve(link, os.path.join(DIR, name))  # download and save image and return success?