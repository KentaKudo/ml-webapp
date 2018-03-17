import os, io, csv, tempfile

from flask import Flask
from flask_script import Manager, Command
from PIL import Image
import numpy as np
import requests
import pickle

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
        token = os.environ['EBAY_TOKEN']
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

class Preprocess(Command):
    def run(self):
        with open("../datasets/jeans.csv", 'r') as file:
            reader = csv.reader(file)
            data = []
            for row in reader:
                image = self.downloadImage(row[1])
                data.append({'x': np.array(image), 'y': float(row[2])})
                if len(data) == 1000:
                    self.save(data)
                    data = []
            if not len(data) == 0:
                self.save(data)

    def downloadImage(self, url):
        buffer = tempfile.SpooledTemporaryFile(max_size=1e9)
        r = requests.get(url, stream=True)
        i = None
        if r.status_code == 200:
            downloaded = 0
            filesize = int(r.headers['content-length'])
            for chunk in r.iter_content():
                downloaded += len(chunk)
                buffer.write(chunk)
            buffer.seek(0)
            i = Image.open(io.BytesIO(buffer.read()))
        buffer.close()
        return i

    def save(self, data):
        datasets = []
        if os.path.exists("../datasets/jeans.pkl"):
            with open('../datasets/jeans.pkl', 'rb') as handle:
                datasets = pickle.load(handle)
        
        with open('../datasets/jeans.pkl', 'wb') as handle:
            pickle.dump(datasets + data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    manager.run({
        'search_jeans': SearchJeans(),
        'preprocess'  : Preprocess()
    })
