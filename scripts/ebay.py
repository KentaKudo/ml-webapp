import os
token = os.environ['EBAY_TOKEN']

from flask import Flask
from flask_script import Manager, Command
import requests

app = Flask(__name__)
manager = Manager(app)

limit = 50
max_amount = 500

class SearchJeans(Command):
    def run(self):
        total = 0
        while True:
            has_next = self.request('https://api.ebay.com/buy/browse/v1/item_summary/search?q=jeans&filter=itemLocationCountry:GB', total)
            if not has_next or total + limit >= max_amount:
                break
            total += limit

    def request(self, url, offset=0):
        r = requests.get(url + "&limit=51&offset="+str(offset), headers={'Authorization': 'Bearer ' + token})        
        response = r.json()
        items = response["itemSummaries"]
        has_next = len(items) > limit
        if has_next:
            items = items[:limit]

        for item in response["itemSummaries"]:
            if "thumbnailImages" in item:
                print(item["itemId"]+","+ \
                      item["thumbnailImages"][0]["imageUrl"]+","+ \
                      item["currentBidPrice"]["convertedFromValue"]+","+ \
                      item["itemWebUrl"])

        return has_next

if __name__ == "__main__":
    manager.run({'search_jeans' : SearchJeans()})

