import data

import os
import time
import shutil
import torch
import platform
import credentials
from utilities import *
from getpass import getpass
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

class Swiper():

    def __init__(self, data_dir, temp_data_dir):

        assert data_dir is not None, "Invalid data folder name"
        assert temp_data_dir is not None, "Invalid temporary data folder name"

        self.data_dir = data_dir
        self.temp_dir = temp_data_dir # to store limbo pictures (while user hasn't swiped right or left)
        self.temp_dir_save = os.path.join(self.temp_dir, "1/")
        self.left_dir = os.path.join(data_dir, "left") # Left category folder, to save left swiped pictures
        self.right_dir = os.path.join(data_dir, "right") # Right category folder

        assert folder_assertions([self.data_dir, self.temp_dir, self.temp_dir_save, self.left_dir, self.right_dir]) == True, "Couldn't create data folders"
        
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
        time.sleep(2)

        # Click on "Login" button
        self.driver.find_element_by_xpath(get_tinder_login_button_xpath()).click()
        time.sleep(1)

        # Click on "Login with Facebook"
        self.driver.find_element_by_xpath(get_tinder_login_with_facebook_xpath()).click()

        print("Ready to start swiping :)")
        return True

    def swipe(self, just_data_extraction=False, auto=True, lib=None):

        """ Extracts and labels pictures from profiles """
        """ just_data_extraction flag swipes left even for profiles you labelled "right" - economizes your likes"""

        libido = lib

        while True:

            # Loop until it finds a profile
            found_profile = False
            done = False


            while not found_profile:
                try:
                    found_profile = self.driver.find_elements_by_xpath(get_current_profile_xpath())
                except NoSuchElementException:
                    try:
                        self.driver(get_add_to_home_button_xpath()).click()
                    except:
                        pass
                    pass

            # Find picture blocks inside current profile
            image_blocks = found_profile[0].find_elements_by_xpath(get_image_blocks_xpath())

            counter = 0
            for current_image_block in image_blocks:

                image = None
                while image is None:
                    try:
                        image = current_image_block.find_element_by_xpath(get_image_container_xpath())
                    except NoSuchElementException:
                        pass


                # Extract current picture
                raw_link = image.get_attribute('style')
                try:
                    save_picture_from_url_block(self.temp_dir_save, raw_link)
                    counter += 1    
                except:
                    print("Couldn't save picture")
                    pass

                # Jump to next picture
                ActionChains(self.driver).send_keys(' ').perform()

            print(f"Saved {counter} pictures")


            # Evaluate
            # TODO: terrible approach, change it
            if auto:
                if libido is not None:
                    evaluation = libido.infer()
                else:
                    c = wait_4_key()
                    evaluation = keys[c]
            else:
                pressed = wait_4_key()
                evaluation = keys[pressed]


            # Get saved pictures list
            image_names = data.get_image_names_in(self.temp_dir_save)
            image_files = [os.path.join(self.temp_dir_save, f) for f in image_names]

            if evaluation == "left":
                
                # TODO: change
                if not auto:
                    print(f"Move pics to left folder")
                    [shutil.move(image, self.left_dir) for image in image_files]

                # Swipe left
                ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                time.sleep(0.5)
                done = True


            elif evaluation == "right":
                
                # TODO: change
                if not auto:
                    print(f"Move pics to right folder")
                    [shutil.move(image, self.right_dir) for image in image_files]

                # If we're just gathering data, we save right swipes in the 'right' folder, but swipe left instead
                if just_data_extraction:
                    
                    # Swipe left
                    ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                    time.sleep(0.5)
                
                else:
                    
                    # Swipe right
                    ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
                    time.sleep(0.5)

                done = True


            for image in image_files:
                os.remove(image)
                print(f"Removed {image}")

                ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                time.sleep(0.5)


        return True