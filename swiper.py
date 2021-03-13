import os
import time
import shutil
import tkinter
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

# TODO: Useful code, delete
# ActionChains(self.driver).send_keys(Keys.ARROW_LEFT).perform()
# ActionChains(self.driver).send_keys(Keys.ARROW_RIGHT).perform()
# time.sleep(0.5)

TMP_IMAGE_DIR = "./__temp__"
DATA_DIR = './data/' # Data directory where your categories folders will be saved
LEFT_DIR = os.path.join(DATA_DIR, "left") # Left category folder
RIGHT_DIR = os.path.join(DATA_DIR, "right") # Right category folder

class Swiper():

    def __init__(self):

        for directory in [TMP_IMAGE_DIR, DATA_DIR, LEFT_DIR, RIGHT_DIR]:
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

    """
    def smart_swipe(self, model, data_transform):

        # Takes prediction model as input
        # Evaluates pictures in Tinder profile
        # and swipes accordingly

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
    """

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