# 
from utilities import *

RAW_URL_BLOCK ='background-image: url("https://images-ssl.gotinder.com/5cb0bdcebce5dd150037ed5a/640x800_226a2e8b-eed7-48a5-9154-ff5ce8c6a744.jpg"); background-position: 50.4587% 0%; background-size: 120.527%;'
DESTINATION = '__test__/'

def test_save_picture_from_url_block(dest, raw):
	a = save_picture_from_url_block(dest, raw)
	print(a)

test_save_picture_from_url_block(DESTINATION, RAW_URL_BLOCK)