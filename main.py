from typing import Union
import requests
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from typing import Any
from fastapi import Body, FastAPI
from starlette.responses import FileResponse
import json

from addlogoTopdf import addLogogo
from test_api_v2 import sendRequestBackend

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 00:41:56 2023

@author: mathi
"""

import math


#  "list_of_pv_area_information": [
#         {"azimuth": 59, "tilt": 38, "name": "Dachanlage 1"}
#     ],

class PV_area_information(BaseModel):
    azimuth: float
    tilt: float
    name: str
    # tag:str ||"tag": "string"
    # tag: str =  Field(examples=["upper_edge","lower_edge"])


class PointC():
    lat: float
    lon: float
    elevation: float
    offset: float


# Class to store points with latitude, longitude, elevation, offset, and a tag
class Point(BaseModel):
    lat: float
    lon: float
    elevation: float
    offset: float


#
# class ReqDataStructure(BaseModel):
#     identifier:int
#     # pv_areas:list[list:BaseModel]]
#     list_of_pv_area_information:list[PV_area_information()]
#     list_of_ops:list[Point(BaseModel)]
#     utc:int
#
#     # def toJson(self):
#     #     return {
#     #         "identifier": self.identifier,  # Hier den Identifier einfügen
#     #         "pv_areas": self.pv_areas.dict()
#     #
#     #         [
#     #             [
#     #                 {"latitude": 48.931985, "longitude": 9.520857, "ground_elevation": 298.75, "height_above_ground": 8.00},
#     #                 {"latitude": 48.932009 	, "longitude": 9.520952, "ground_elevation": 298.75, "height_above_ground": 4.10},
#     #                 {"latitude": 48.932090, "longitude": 9.520873, "ground_elevation": 298.75, "height_above_ground": 4.10},
#     #                 {"latitude": 48.932063, "longitude": 9.520780, "ground_elevation": 298.75, "height_above_ground": 8.00}
#     #             ]
#     #         ],
#     #         "list_of_pv_area_information": [
#     #             {"azimuth": 59, "tilt": 38, "name": "Dachanlage 1"}
#     #         ],
#     #         "list_of_ops": [
#     #             {"latitude": 48.932100, "longitude": 9.521008, "ground_elevation": 301.00, "height_above_ground": 1.50}
#     #         ],
#     #         "utc": 1
#     #     }
#
#     # tag:str ||"tag": "string"
#     # tag: str =  Field(examples=["upper_edge","lower_edge"])
#


# Function to calculate azimuth angle between two coordinates
def calculate_azimuth(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    lon1 = math.radians(lon1)
    lon2 = math.radians(lon2)

    d_lon = lon2 - lon1

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(d_lon))

    azimuth = math.atan2(x, y)
    azimuth = math.degrees(azimuth)
    azimuth = (azimuth + 360) % 360

    return azimuth


# Function to calculate the PV azimuth
def calculate_pv_azimuth(points):
    # Find points with the tag "upper_edge"
    upper_edges = [p for p in points if p.tag == "upper_edge"]

    # Ensure exactly two "upper_edge" points exist
    if len(upper_edges) != 2:
        raise ValueError("There must be exactly two points with the tag 'upper_edge'. and two with 'lower_edge' ")

    # Calculate the azimuth angle between the two "upper_edge" points
    edge_orientation = calculate_azimuth(upper_edges[0].lat, upper_edges[0].lon,
                                         upper_edges[1].lat, upper_edges[1].lon)

    # Calculate azimuths to the other points
    azimuths_to_other_points = []
    for point in points:
        if point.tag != "upper_edge":
            azimuth = calculate_azimuth(upper_edges[0].lat, upper_edges[0].lon, point.lat, point.lon)
            azimuths_to_other_points.append(azimuth)

    # Check if both azimuths are within the range of edge_orientation - 90°
    # (in code it is 95 to have a bit error tolerance if we dont have perfect 90° angles)
    # here it is important that the user is only allowed to draw rectangles (or lines)
    for azimuth in azimuths_to_other_points:
        diff = (azimuth - (edge_orientation - 95)) % 360
        if not (0 <= diff <= 180):
            return (edge_orientation + 90) % 360

    return (edge_orientation - 90) % 360


# Example coordinates forming a rectangle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Solar": "Glare V 1.0"}


# @app.put("/items/{item_id}")
# async def update_item(item_id: int, item: Item, user: User):
#     results = {"item_id": item_id, "item": item, "user": user}
#     return results

@app.put("/azimuth/")
async def update_item(point1: Point, point2: Point, point3: Point, point4: Point):
    # point1 = Point(53.61591272, 9.98706581, 0, 0, "upper_edge")
    # point2 = Point(53.61589630, 9.98716121, 0, 0, "upper_edge")
    # point3 = Point(53.61583476, 9.98712882, 0, 0, "lower_edge")
    # point4 = Point(53.61585252, 9.98703871, 0, 0, "lower_edge")

    points = [point1, point2, point3, point4]

    # Calculate the PV Azimuth with the provided points
    try:
        pv_azimuth = calculate_pv_azimuth(points)
    except  Exception as e:
        print(e)
        return {"error": str(e), }

    return {"aziumth": pv_azimuth, }


@app.get("/elevation/")
async def update_item(lat: float, long: float):
    api_key = os.environ.get("API_KEY", )

    reqUrl = "https://maps.googleapis.com/maps/api/elevation/json?locations=" + str(lat) + "%2C" + str(long) + "&key=" + api_key
    print(reqUrl)
    response = requests.get(reqUrl)

    # Calculate the PV Azimuth with the provided points

    return response.json(),


@app.post('/getPDF')
async def getPDF(payload: Any = Body(None)):
    """
    Create an item with all the information:

    - **name**: each item must have a name
    - **description**: a long description
    - **price**: required
    - **tax**: if the item doesn't have tax, you can omit this

    Example:

    ```
    {
        "identifier":123,
         "pv_areas": [
             [
                 {"latitude": 48.931985, "longitude": 9.520857, "ground_elevation": 298.75, "height_above_ground": 8.00},
                 {"latitude": 48.932009 	, "longitude": 9.520952, "ground_elevation": 298.75, "height_above_ground": 4.10},
                 {"latitude": 48.932090, "longitude": 9.520873, "ground_elevation": 298.75, "height_above_ground": 4.10},
                 {"latitude": 48.932063, "longitude": 9.520780, "ground_elevation": 298.75, "height_above_ground": 8.00}
             ]
         ],
         "list_of_pv_area_information": [
             {"azimuth": 59, "tilt": 38, "name": "Dachanlage 1"}
         ],
         "list_of_ops": [
             {"latitude": 48.932100, "longitude": 9.521008, "ground_elevation": 301.00, "height_above_ground": 1.50}
         ],
         "utc": 1
     }
    ```

    """

    # return FileResponse('assets/report.pdf', media_type='application/octet-stream', filename='report.pdf')

    sendRequestBackend(payload)
    file_name = str(payload['identifier'])
    utc = str(payload['utc'])
    file_path = os.getcwd() + "/assets/" + file_name + '.pdf'

    addLogogo(file_path,utc)

    print(payload)

    return FileResponse(file_path, media_type='application/octet-stream', filename=file_name + '.pdf')

# return payload
