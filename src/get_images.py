import requests
import time

def main():
  with open("debug/urls_txt.txt", "r") as f:
    data = f.readlines()
    urls = data[-1000:]
    # import ipdb; ipdb.set_trace()

    for url in urls:
      id, url = url.split("\t")
      url = url.replace("\n", "")
      params = {'User-Agent' : "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.84 Safari/537.36",
                "Referer" : url}
      response = requests.get(url, headers=params)
      with open("validations/" + id + ".jpg", "wb") as f:
        f.write(response.content)
      
      time.sleep(2)
      print(id + ", done!")

if __name__ == "__main__":
  main()