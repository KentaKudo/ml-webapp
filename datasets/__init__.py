import os
import pickle, random
import numpy as np
from sklearn.model_selection import train_test_split

# http://pages.ebay.com/sellerinformation/growing/categorychanges/clothing-all.html
categories = [
    {'name': 'casual_shirts',          'id': 57990},
    {'name': 'dress_shirts',           'id': 57991},
    {'name': 't_shirts',               'id': 15687},
    {'name': 'athletic_apparel',       'id': 137084},
    {'name': 'blazers_and_sport_oats', 'id': 3002},
    {'name': 'coats_and_jackets',      'id': 57988},
    {'name': 'jeans',                  'id': 11483},
    {'name': 'pants',                  'id': 57989},
    {'name': 'shorts',                 'id': 15689},
    {'name': 'sleepwear_and_robes',    'id': 11510},
    {'name': 'socks',                  'id': 11511},
    {'name': 'suits',                  'id': 3001},
    {'name': 'sweaters',               'id': 11484},
    {'name': 'sweats_and_hoodies',     'id': 155183},
    {'name': 'swimwear',               'id': 15690},
    {'name': 'underwear',              'id': 1507},
    {'name': 'vests',                  'id': 15691},
    {'name': 'mixed_items_and_lots',   'id': 84434},
]
num_classes = len(categories)

def load_datasets():
  with open(os.path.dirname(__file__)+"/datasets.pkl", 'rb') as f:
    datasets = pickle.load(f)
    random.shuffle(datasets)
    X = [v['x'] for v in datasets]
    y = [v['y'] for v in datasets]
    (X_train, X_test, y_train, y_test) = train_test_split(np.array(X),
                                                          np.array(y),
                                                          test_size=0.2,
                                                          random_state=42)
  return (X_train, y_train), (X_test, y_test)
