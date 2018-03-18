import os, io, csv, tempfile
import sys
sys.path.append('..')

import numpy as np
import requests
import pickle
from flask import Flask
from flask_script import Manager, Command
from PIL import Image
from utils import resizeImage

app = Flask(__name__)
manager = Manager(app)

limit = 200
max_per_cat = 10000
base_url = "https://api.ebay.com/buy/browse/v1"
endpoint = "/item_summary/search"

# http://pages.ebay.com/sellerinformation/growing/categorychanges/clothing-all.html
categories = {
    'casual_shirts'         : 57990,
    'dress_shirts'          : 57991,
    't_shirts'              : 15687,
    'athletic_apparel'      : 137084,
    'blazers_and_sport_oats': 3002,
    'coats_and_jackets'     : 57988,
    'jeans'                 : 11483,
    'pants'                 : 57989,
    'shorts'                : 15689,
    'sleepwear_and_robes'   : 11510,
    'socks'                 : 11511,
    'suits'                 : 3001,
    'sweaters'              : 11484,
    'sweats_and_hoodies'    : 155183,
    'swimwear'              : 15690,
    'underwear'             : 1507,
    'vests'                 : 15691,
    'mixed_items_and_lots'  : 84434,
}

class Mine(Command):
    def run(self):
        for k, v in categories.items():
            total = 0
            url = base_url+endpoint+"?category_ids="+str(v)+"&limit="+str(limit)
            with open("../datasets/"+k+".csv", 'w') as file:
                writer = csv.writer(file)
                while total < max_per_cat and url is not None:
                    url, items = self.request(url)
                    writer.writerows(items)
                    total += len(items)

    def request(self, url):
        token = os.environ['EBAY_TOKEN']
        r = requests.get(url, headers={'Authorization: Bearer '+token})
        response = r.json()
        if not "itemSummaries" in response:
            return response["next"] if "next" in response else None, []
        items = list(map(lambda x: [x["itemId"], x["image"]["imageUrl"]],
                         filter(lambda x: "image" in item, response["itemSummaries"])))
        return response["next"] if "next" in response else None, items

class Preprocess(Command):
    def run(self):
        with open("../datasets/jeans.csv", 'r') as file:
            reader = csv.reader(file)
            data = []
            for row in reader:
                image = self.downloadImage(row[1])
                resized = resizeImage(image, target_size=(225, 225))
                data.append({'x': np.array(resized), 'y': float(row[2])})
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
