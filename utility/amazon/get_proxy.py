import random
import requests

def getRandomProxy():
    print("proxy")
    # proxies = [
    #     "http://wyrpqchj-rotate:4xfau0yx5zy5@p.webshare.io:80/",
    #     "http://wyrpqchj-rotate:4xfau0yx5zy5@p.webshare.io:80/"
    # ]

    # proxy = random.choice(proxies)

    # try:
    #     response = requests.get("https://ipv4.webshare.io/", proxies={"http": proxy, "https": proxy})
    #     response.raise_for_status()
    #     ip_address = response.text.strip()
    #     return {
    #         "status": "OK",
    #         "reason": "",
    #         "data": {
    #             "carrier": "",
    #             "city": "London",
    #             "country_code": "UR",
    #             "country_name": "America",
    #             "ip": ip_address,
    #             "isp": "FORTHnet",
    #             "region": "Thesaly"
    #         }
    #     }
    # except requests.RequestException as e:
    #     return {
    #         "status": "ERROR",
    #         "reason": str(e),
    #         "data": {}
    #     }