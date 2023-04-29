from selectorlib import Extractor
import requests 
from dateutil import parser as dateparser
from utility.amazon.get_proxy import getRandomProxy
from selenium import webdriver

# driver_path = '/home/mhakaal10/Downloads/chromedriver_linux64 (1)/chromedriver'
# browser = webdriver.Chrome(executable_path=driver_path)
# options = webdriver.ChromeOptions()
# options.add_argument('--disable-blink-features=AutomationControlled')
# options.add_argument('--headless')



# Create an Extractor by reading from the YAML file
e = Extractor.from_yaml_file('utility/amazon/selectors.yml')

def scrape(url):    

   
    headers = {
        'dnt': '1',
        'upgrade-insecure-requests': '1',
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.61 Safari/537.36',
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'sec-fetch-site': 'same-origin',
        'sec-fetch-mode': 'navigate',
        'sec-fetch-user': '?1',
        'sec-fetch-dest': 'document',
        'referer': 'https://www.amazon.com/',
        'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    }

    proxy = requests.get(
    "https://ipv4.webshare.io/",
    proxies={
        "http": "http://wyrpqchj-rotate:4xfau0yx5zy5@p.webshare.io:80/",
        "https": "http://wyrpqchj-rotate:4xfau0yx5zy5@p.webshare.io:80/"
        }
    ).text
    prox = {
            "status": "OK",
            "reason": "",
            "data": {
                "carrier": "",
                "city": "London",
                "country_code": "UR",
                "country_name": "America",
                "ip": proxy,
                "isp": "FORTHnet",
                "region": "Thesaly"
            }
        }
    
    print(f'proxies {proxy}')
    # solve_captcha(url)
    # Download the page using requests
    print("Downloading %s"%url)
    r = requests.get(url, headers=headers, proxies=prox)
    # Simple check to check if page was blocked (Usually 503)
    if r.status_code > 500:
        if "To discuss automated access to Amazon data please contact" in r.text:
            print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
        else:
            print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
        return None
    # Pass the HTML of the page and create 

    return e.extract(r.text)


def scrapped(url):
    reviewList = []
    data = scrape(url) 
    # browser.get(url) 

    print(f'data response {data}')

    if data:
        print(reviewList)
        for r in data['reviews']:
            r["product"] = data["product_title"]
            r['url'] = url
            if 'verified' in r:
                if 'Verified Purchase' in r['verified']:
                    r['verified'] = 'Yes'
                else:
                    r['verified'] = 'No'
            r['rating'] = r['rating'].split(' out of')[0]
            date_posted = r['date'].split('on ')[-1]
            if r['images']:
                r['images'] = "\n".join(r['images'])
            r['date'] = dateparser.parse(date_posted).strftime('%d %b %Y')
            reviewList.append(r)
    
    return reviewList

