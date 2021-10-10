import requests
from pygeodesy.ellipsoidalVincenty import LatLon
import datetime
from PIL import Image
from io import BytesIO
import numpy as np
import json
import math
import asyncio


model_server_url = 'http://localhost:8501/v1/models/raster_dtfr:predict'


def get_image(coord):
    pass


def bbox_along_geometry(coords):
    coords = (np.array(coords) / 10 ** 7).tolist()
    hop_len = 5
    list_gen_locs = []
    list_bearings = []
    for i in range(len(coords) - 1):
        coord1 = LatLon(coords[i][1], coords[i][0])
        coord2 = LatLon(coords[i + 1][1], coords[i + 1][0])
        bearing = coord1.compassAngleTo(coord2)
        dist = coord1.distanceTo(coord2)
        init_loc = coord1
        next_loc = init_loc.destination(hop_len, bearing)
        dist_from_coord1 = coord1.distanceTo(next_loc)
        while dist_from_coord1 < dist:
            list_gen_locs.append([next_loc.lon, next_loc.lat])
            list_bearings.append(bearing)
            init_loc = next_loc
            next_loc = init_loc.destination(hop_len, bearing)
            dist_from_coord1 = coord1.distanceTo(next_loc)
    mid = (len(list_gen_locs)) // 2
    if len(list_gen_locs) == 0:
        return None, None
    return list_gen_locs[mid], list_bearings[mid]


def predict(img):
    data = np.expand_dims(img / 255., axis=0)
    data = data.tolist()
    data = json.dumps({"instances": data})
    headers = {"content-type": "application/json"}
    json_response = requests.post(model_server_url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions
