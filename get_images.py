import requests
import hashlib
from PIL import Image
import csv

class PixivImageRequest(object):
    def __init__(self):
        self.urls = self.load_urls()

    def load_urls(self):
        with open("urls.csv", "r") as f:
            self.urls = [row[0] for row in csv.reader(f)]

    def request(self):
        for url in self.urls:
            try:
                response = requests.get(url, headers={'referer':url, "User-Agent":"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36"})
                yield response.content
            except:
                print("can't get the image")
                continue


if __name__ == "__main__":
    p_request = PixivImageRequest()
    p_request.load_urls()

    for i, binary in enumerate(p_request.request()):
        with open("images/" + str(i) + ".jpg", "wb") as f:
            f.write(binary)
        print("just {} data loaded".format(i))
