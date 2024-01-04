from typing import Union

from fastapi import FastAPI

from pydantic import BaseModel

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 00:41:56 2023

@author: mathi
"""

import math

# Class to store points with latitude, longitude, elevation, offset, and a tag
class Point(BaseModel):
    lat:float
    lon:float
    elevation:float
    offset:float
    tag:str

    # def __init__(self, lat, lon, elevation, offset, tag):
    #     super().__init__(lat=lat, damage=damage)
    #     self.lat = lat
    #     self.lon = lon
    #     self.elevation = elevation # ground elevation
    #     self.offset = offset # height above ground elevateion
    #     self.tag = tag # the tag is for upper_edge oder lower_edge

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
        raise ValueError("There must be exactly two points with the tag 'upper_edge'.")

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


@app.get("/")
def read_root():
    return {"Solar": "Glare V 1.0"}




# @app.put("/items/{item_id}")
# async def update_item(item_id: int, item: Item, user: User):
#     results = {"item_id": item_id, "item": item, "user": user}
#     return results

@app.put("/azimuth/")
async def update_item( point1: Point, point2: Point,point3: Point,point4: Point ):

    # point1 = Point(53.61591272, 9.98706581, 0, 0, "upper_edge")
    # point2 = Point(53.61589630, 9.98716121, 0, 0, "upper_edge")
    # point3 = Point(53.61583476, 9.98712882, 0, 0, "lower_edge")
    # point4 = Point(53.61585252, 9.98703871, 0, 0, "lower_edge")

    points = [point1, point2, point3, point4]

    # Calculate the PV Azimuth with the provided points
    pv_azimuth = calculate_pv_azimuth(points)

    return {"aziumth": pv_azimuth, }