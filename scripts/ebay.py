import os, csv
token = os.environ['EBAY_TOKEN']

from flask import Flask
from flask_script import Manager, Command
import requests

app = Flask(__name__)
manager = Manager(app)

max_amount = 50000

class SearchJeans(Command):
    def run(self):
        url = "https://api.ebay.com/buy/browse/v1/item_summary/search?q=jeans&filter=itemLocationCountry:GB&limit=200"
        total = 0
        with open("../datasets/jeans.csv", 'a') as file:
            writer = csv.writer(file)
            while True:
                url, items = self.request(url)
                writer.writerows(items)
                if url is None or total + len(items) >= max_amount:
                    print(url, total, len(items))
                    break
                total += len(items)

    def request(self, url):
        r = requests.get(url, headers={'Authorization': 'Bearer ' + token})        
        response = r.json()
        if "warnings" in response and "message" in response["warnings"]:
            print(response["warnings"]["message"])
        if not "itemSummaries" in response:
            return response["next"] if "next" in response else None, []
        items = list(map(lambda item: [item["itemId"],
                                       item["image"]["imageUrl"],
                                       item["price"]["value"]],
                         filter(lambda item: "image" in item, response["itemSummaries"])))

        return response["next"] if "next" in response else None, items

if __name__ == "__main__":
    manager.run({'search_jeans' : SearchJeans()})
