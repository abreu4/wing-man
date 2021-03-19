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

    def __init__(self, data_dir):

        if not os.path.isdir(data_dir): os.mkdir(data_dir)

        TMP_IMAGE_DIR = "./__temp__" # to store limbo pictures (while user hasn't swiped right or left)
        LEFT_DIR = os.path.join(data_dir, "left") # Left category folder, to save left swiped pictures
        RIGHT_DIR = os.path.join(data_dir, "right") # Right category folder

        for directory in [TMP_IMAGE_DIR, LEFT_DIR, RIGHT_DIR]:
            if not os.path.exists(directory):
                os.mkdir(directory)
                print(f"Created {directory} successfully")
        
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

    def manual_swipe(self, just_data_extraction=False):

        """ Extracts and labels pictures from profiles """
        """ just_data_extraction flag swipes left even for profiles you labelled "right" - economizes your likes"""

        while True:

            # Loop until it finds a profile
            found_profile = False
            done = False

            while not found_profile:
                try:
                    found_profile = self.driver.find_elements_by_xpath(get_current_profile_xpath())
                except NoSuchElementException:
                    pass

            print(f"Found a profile: {found_profile}")

            # Find picture blocks inside current profile
            image_blocks = found_profile[0].find_elements_by_xpath(get_image_blocks_xpath())

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
                    save_picture_from_url_block(TMP_IMAGE_DIR, raw_link)
                    print(f"Saved picture at {TMP_IMAGE_DIR}")
                except:
                    print("Couldn't save picture")

                # Jump to next picture
                ActionChains(self.driver).send_keys(' ').perform()


            image_files = [os.path.join(TMP_IMAGE_DIR, f) for f in os.listdir(TMP_IMAGE_DIR) if os.path.isfile(os.path.join(TMP_IMAGE_DIR, f)) and (f.endswith('.jpg') or f.endswith('.webp'))]
            print(f"Saved {len(image_files)} pictures")

            while not done:

                c = wait_4_key()

                try:
                    pressed = keys[c]

                    # Press 1 to swipe left
                    if keys[c] == "left":
                        
                        print(f"Move pics to left folder")

                        [shutil.move(image, LEFT_DIR) for image in image_files]

                        # Swipe left
                        ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                        time.sleep(0.5)
                        done = True


                    # Press 2 to swipe right
                    elif keys[c] == "right":
                        
                        print(f"Move pics to right folder")

                        [shutil.move(image, RIGHT_DIR) for image in image_files]

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

                # Press any other key to ignore 
                except KeyError:
                    for image in image_files:
                            print(f"Removed {image}")
                            os.remove(image)

                    ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
                    time.sleep(0.5)

                    done = True



        return 1