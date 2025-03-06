import os
import json
import re
import random
from utils import Utils
from element_class import Window, Wall, Stair, Roof, Ramp, Floor, Door, Ceiling, Column
from tqdm import tqdm

def create_window():
    name = "window"
    generation = ["Level", "Sill Height", "Orientation", "Head Height", "Comments", "Wall Closure", "Glass Pane Material", "Glass Pane Color", "Sash Material", "Sash Color", "Height", "Width", "Window Inset",  "Type Comments", ]
    retrieval = ["Level", "Sill Height", "Orientation", "Head Height", "Comments", "Wall Closure", "Glass Pane Material", "Glass Pane Color", "Sash Material", "Sash Color", "Height", "Width", "Window Inset", "Type Comments", "Coordinate"]
    window = Window(name, generation, retrieval)
    return window

def create_wall():
    name = "wall"
    generation = ["Base Constraint", "Base Offset", "Top Constraint", "Top Offset", "Unconnected Height", "Room Bounding", "Cross-Section", "Angle from Vertical", "Structural", "Structural Usage", "Comments", "Wrapping at Inserts", "Wrapping at Ends", "Width", "Function", "Material", "Color", "Type Comments"]
    retrieval = ["Base Constraint", "Base Offset", "Top Constraint", "Top Offset", "Unconnected Height", "Room Bounding", "Cross-Section", "Angle from Vertical", "Structural", "Structural Usage", "Comments", "Wrapping at Inserts", "Wrapping at Ends", "Width", "Function", "Material", "Color", "Type Comments", "Length", "Area", "Volume", "Coordinate"]
    wall = Wall(name, generation, retrieval)
    return wall

def create_stair():
    name = "stair"
    generation = ["Base Level", "Base Offset", "Top Level", "Top Offset", "Desired Stair Height", "Desired Number of Risers", "Actual Tread Depth", "Comments", "Maximum Riser Height", "Minimum Tread Depth", "Minimum Run Width", "Function", "Right Support", "Right Lateral Offset", "Left Support", "Left Lateral Offset", "Middle Support", "Middle Support Number", "Type Comments"]
    retrieval = ["Base Level", "Base Offset", "Top Level", "Top Offset", "Desired Stair Height", "Desired Number of Risers", "Actual Tread Depth", "Comments", "Maximum Riser Height", "Minimum Tread Depth", "Minimum Run Width", "Function", "Right Support", "Right Lateral Offset", "Left Support", "Left Lateral Offset", "Middle Support", "Middle Support Number", "Type Comments", "Actual Number of Risers", "Actual Riser Height", "Coordinate"]
    stair = Stair(name, generation, retrieval)
    return stair

def create_roof():
    name = "roof"
    generation = ["Base Level", "Room Bounding", "Base Offset From Level", "Rafter Cut", "Fascia Depth", "Rafter or Truss", "Slope", "Comments", "Material", "Color", "Thickness", "Type Comments"]
    retrieval = ["Base Level", "Room Bounding", "Base Offset From Level", "Rafter Cut", "Fascia Depth", "Rafter or Truss", "Slope", "Comments", "Material", "Color", "Thickness", "Type Comments", "Maximum Ridge Height", "Volume", "Area", "Coordinate"]
    roof = Roof(name, generation, retrieval)
    return roof

def create_ramp():
    name = "ramp"
    generation = ["Base Level", "Base Offset", "Top Level", "Top Offset", "Multistory Top Level", "Width", "Comments", "Shape", "Thickness", "Function", "Material", "Color", "Maximum Incline Length", "Max Slope", "Type Comments"]
    retrieval = ["Base Level", "Base Offset", "Top Level", "Top Offset", "Multistory Top Level", "Width", "Comments", "Shape", "Thickness", "Function", "Material", "Color", "Maximum Incline Length", "Max Slope", "Type Comments", "Coordinate"]
    ramp = Ramp(name, generation, retrieval)
    return ramp

def create_floor():
    name = "floor"
    generation = ["Level", "Height Offset from Level", "Room Bounding", "Structural", "Slope", "Comments", "Material", "Color", "Thickness", "Function", "Type Comments"]
    retrieval = ["Level", "Height Offset from Level", "Room Bounding", "Structural", "Slope", "Comments", "Material", "Color", "Thickness", "Function", "Type Comments", "Perimeter", "Area", "Volume", "Coordinate"]
    floor = Floor(name, generation, retrieval)
    return floor

def create_door():
    name = "door"
    generation = ["Level", "Sill Height", "Orientation", "Head Height", "Comments", "Wall Closure", "Function", "Door Material", "Door Color", "Frame Material", "Frame Color", "Thickness", "Height", "Trim Projection Exterior", "Trim Projection Interior", "Trim Width", "Width", "Type Comments"]
    retrieval = ["Level", "Sill Height", "Orientation", "Head Height", "Comments", "Wall Closure", "Function", "Door Material", "Door Color", "Frame Material", "Frame Color", "Thickness", "Height", "Trim Projection Exterior", "Trim Projection Interior", "Trim Width", "Width", "Type Comments", "Coordinate"]
    door = Door(name, generation, retrieval)
    return door

def create_ceiling():
    name = "ceiling"
    generation = ["Level", "Height Offset from Level", "Room Bounding", "Slope", "Comments", "Material", "Color", "Thickness", "Type Comments"]
    retrieval = ["Level", "Height Offset from Level", "Room Bounding", "Slope", "Comments", "Material", "Color", "Thickness", "Type Comments", "Perimeter", "Area", "Volume", "Coordinate"]
    ceiling = Ceiling(name, generation, retrieval)
    return ceiling

def create_column():
    name = "column"
    generation = ["Base Level", "Base Offset", "Top Level", "Top Offset", "Moves with Grids", "Room Bounding", "Comments", "Material", "Color", "Depth", "Offset Base", "Offset Top", "Width", "Type Comments"]
    retrieval = ["Base Level", "Base Offset", "Top Level", "Top Offset", "Moves with Grids", "Room Bounding", "Comments", "Material", "Color", "Depth", "Offset Base", "Offset Top", "Width", "Type Comments", "Coordinate"]
    column = Column(name, generation, retrieval)
    return column

def generate_instances():
    window = create_window()
    wall = create_wall()
    stair = create_stair()
    roof = create_roof()
    ramp = create_ramp()
    floor = create_floor()
    door = create_door()
    ceiling = create_ceiling()
    column = create_column()

    num = 600
    all_elements = [stair]
    names = ["stair"]
    results_file = "../data/origin/{}.json"

    for name, element in zip(names, all_elements):
        all_data = []
        for i in tqdm(range(num)):
            all_data.append(element.generate())
        with open(results_file.format(name), "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4)

if __name__ == "__main__":
    generate_instances()
