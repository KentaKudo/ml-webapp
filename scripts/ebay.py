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
from tqdm import tqdm
from datasets import categories
from models import image_size

app = Flask(__name__)
manager = Manager(app)

limit = 200
max_per_cat = 10000
base_url = "https://api.ebay.com/buy/browse/v1"
endpoint = "/item_summary/search"
batch_size = 1000
pkl_per_cat = int(os.environ['PER_CAT']) if 'PER_CAT' in os.environ else 3000

class Mine(Command):
    def run(self):
        for v in tqdm(categories):
            total = 0
            url = base_url+endpoint+"?category_ids="+str(v['id'])+"&limit="+str(limit)
            with open("../datasets/"+v['name']+".csv", 'w') as file:
                writer = csv.writer(file)
                while total < max_per_cat and url is not None:
                    url, items = self.request(url)
                    writer.writerows(items)
                    total += len(items)

    def request(self, url):
        token = os.environ['EBAY_TOKEN']
        r = requests.get(url, headers={'Authorization': 'Bearer '+token})
        response = r.json()
        if not "itemSummaries" in response:
            return response["next"] if "next" in response else None, []
        items = list(map(lambda x: [x["itemId"], x["image"]["imageUrl"]],
                         filter(lambda x: "image" in x, response["itemSummaries"])))
        return response["next"] if "next" in response else None, items

class Pickle(Command):
    def run(self):
        for i, v in enumerate(tqdm(categories)):
            with open("../datasets/"+v['name']+".csv", "r") as file:
                reader = csv.reader(file)
                data = []
                count = 0
                for row in reader:
                    image = self.downloadImage(row[1]).convert('RGB')
                    resized = resizeImage(image, target_size=image_size)
                    data.append({'x': np.array(resized), 'y': i})
                    if len(data) >= batch_size:
                        self.save(data)
                        data = []
                    count += 1
                    if count >= pkl_per_cat:
                        break
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
        if os.path.exists("../datasets/datasets.pkl"):
            with open('../datasets/datasets.pkl', 'rb') as handle:
                datasets = pickle.load(handle)
        
        with open('../datasets/datasets.pkl', 'wb') as handle:
            pickle.dump(datasets + data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    manager.run({
        'mine'  : Mine(),
        'pickle': Pickle(),
    })
