from bs4 import BeautifulSoup
import requests
from urllib.request import *
from urllib.parse import *
from os import makedirs
import os.path, time, re

def crawler():
    # url = 'https://www.vgmusic.com/music/console/nintendo/gameboy/'
    # url = 'https://freemidi.org/artist-1586-lady-gaga'
    url = 'https://freemidi.org/artist-736-michael-jackson'
    html = requests.get(url)
    soup = BeautifulSoup(html.text, 'html.parser')
    links = soup.select('div > span > a[href]')
    total_links = []
    for link in links:
        total_links.append(url + link['href'])

    counter = 0
    for each in total_links:
        try:
            urlretrieve(each, '/data/pop/'+each.replace(url+'download3-', '').lstrip('1234567890-'))
        except:
            print('ERROR: ' + each)
        finally:
            counter += 1
            print(str(counter) + "downloaded: "+ each)



crawler()
