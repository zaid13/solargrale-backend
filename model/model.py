from pydantic import BaseModel, Field

class PV_area_information(BaseModel):
    azimuth: float
    tilt: float
    name: str
    # tag:str ||"tag": "string"
    # tag: str =  Field(examples=["upper_edge","lower_edge"])




# Class to store points with latitude, longitude, elevation, offset, and a tag
class Point(BaseModel):
    lat: float
    lon: float
    elevation: float
    offset: float
    tag: str


class Simulation_Parameter(BaseModel):
    grid_width: float
    resolution: str
    sun_elevation_threshold: int
    beam_spread: float
    sun_angle: float
    sun_reflection_threshold: float
    zoom_level: int


class MetaData(BaseModel):
    user_id: str
    project_id: str
    sim_id: str
    timestamp: int
    utc: int
    project_name: str

class GlareRequestModel(BaseModel):
    identifier: str
    pv_areas: list
    list_of_pv_area_information: list
    list_of_ops: list
    excluded_areas: list
    meta_data: MetaData
    simulation_parameter: Simulation_Parameter


