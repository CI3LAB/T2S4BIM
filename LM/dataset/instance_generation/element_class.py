import os
import json
import re
import random
from utils import Utils

class Element():
    def __init__(self, name, generation, retrieval):
        self.name = name
        self.generation = generation
        self.retrieval = retrieval
        self.max_num_gen = 6
        self.intents = ["creation", "deletion", "retrieval", "modification"]
        self.intents_weights = [0.2, 0.05, 0.15, 0.6]
        self.ranges = {}
        self.utils = Utils()
    
    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()
    
    def generate_creation(self):
        with open("../structured_output_template/template.txt", "r") as f:
            template = f.read()
        template = template.replace("[ELEMENT]", self.name)
        template = template.replace("[INTENT]", "creation")
        slots_str = ""
        num_gen = random.randint(1, self.max_num_gen)
        property_list = random.sample(self.generation, num_gen)
        for property in property_list:
            if property in self.utils.option_based_property:
                value_range = self.ranges[property]
                value = random.choice(value_range)
                str = "(property: {}\nvalue: {})\n".format(property, value)
            elif property in self.utils.scalar_based_property:
                value_range = self.ranges[property]
                if property in self.utils.degree_based_property:
                    value, unit = self.utils.get_num_degree(value_range[0], value_range[1])
                elif property in self.utils.no_unit_scalar_based_property:
                    value = random.randint(value_range[0], value_range[1])
                    unit = "N/A"
                else:
                    value, unit = self.utils.get_num(value_range[0], value_range[1])
                str = "(property: {}\nvalue: {}\nunit: {})\n".format(property, value, unit)
            else: # property in text_based_property
                if property not in self.utils.text_based_property:
                    print("WRONG!!!")
                value = self.utils.generate_comment(property, self.name)
                str = "(property: {}\nvalue: {})\n".format(property, value)
            slots_str += str
        template = template.replace("[SLOTS]\n", slots_str)
        return template
    
    def generate_deletion(self):
        with open("../structured_output_template/template.txt", "r") as f:
            template = f.read()
        template = template.replace("[ELEMENT]", self.name)
        template = template.replace("[INTENT]", "deletion")
        template = template.replace("[SLOTS]\n", "")
        return template
    
    def generate_retrieval(self):
        with open("../structured_output_template/template.txt", "r") as f:
            template = f.read()
        template = template.replace("[ELEMENT]", self.name)
        template = template.replace("[INTENT]", "retrieval")
        slots_str = ""
        num_gen = random.randint(1, self.max_num_gen)
        property_list = random.sample(self.retrieval, num_gen)
        for property in property_list:
            str = "(property: {})\n".format(property)
            slots_str += str
        template = template.replace("[SLOTS]\n", slots_str)
        return template
    
    def generate_modification(self):
        with open("../structured_output_template/template.txt", "r") as f:
            template = f.read()
        template = template.replace("[ELEMENT]", self.name)
        template = template.replace("[INTENT]", "modification")
        slots_str = ""
        num_weights = list(range(self.max_num_gen, 0, -1))
        num_list = list(range(1, self.max_num_gen + 1))
        num_gen = random.choices(num_list, weights=num_weights)[0]
        property_list = random.sample(self.generation, num_gen)
        for property in property_list:
            if property in self.utils.option_based_property:
                value_range = self.ranges[property]
                from_e, to_e = self.utils.get_from_to(value_range)
                if random.random() < 0.5:
                    str = "(property: {}\ntarget_value: {})\n".format(property, to_e)
                else:
                    str = "(property: {}\nsource_value: {}\ntarget_value: {})\n".format(property, from_e, to_e)
            elif property in self.utils.scalar_based_property:
                value_range = self.ranges[property]
                if property in self.utils.degree_based_property:
                    from_n, unit_1, to_n, unit_2 = self.utils.get_two_nums_degree(value_range[0], value_range[1])
                elif property in self.utils.no_unit_scalar_based_property:
                    from_n = random.randint(value_range[0], value_range[1])
                    to_n = random.randint(value_range[0], value_range[1])
                    while to_n == from_n:
                        to_n = random.randint(value_range[0], value_range[1])
                    unit_1 = "N/A"
                    unit_2 = "N/A"
                else:
                    from_n, unit_1, to_n, unit_2 = self.utils.get_two_nums(value_range[0], value_range[1])
                if random.random() < 0.2:
                    to_n, unit_2 = self.utils.get_times()
                if random.random() < 0.5:
                    str = "(property: {}\nsource_value: {}\nsource_unit: {}\ntarget_value: {}\ntarget_unit: {})\n".format(property, from_n, unit_1, to_n, unit_2)
                else:
                    str = "(property: {}\ntarget_value: {}\ntarget_unit: {})\n".format(property, to_n, unit_2)
            else: # property in text_based_property
                if property not in self.utils.text_based_property:
                    print("WRONG!!!")
                from_t = self.utils.generate_comment(property, self.name)
                to_t = self.utils.generate_comment(property, self.name)
                if random.random() < 0.5:
                    str = "(property: {}\nsource_value: {}\ntarget_value: {})\n".format(property, from_t, to_t)
                else:
                    str = "(property: {}\ntarget_value: {})\n".format(property, to_t)
            slots_str += str
        template = template.replace("[SLOTS]\n", slots_str)
        return template

class Window(Element):
    def __init__(self, name, generation, retrieval):
        super(Window, self).__init__(name, generation, retrieval)
        self.level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.sill_height = [-10000, 10000]
        self.orientation = ["Vertical", "Slanted"]
        self.head_height = [-10000, 10000]
        self.wall_closure = ["By host", "Neither", "Interior", "Exterior", "Both"]
        self.glass_pane_material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.glass_pane_color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.sash_material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.sash_color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.height = [0, 10000]
        self.width = [0, 10000]
        self.window_inset = [0, 10000]
        self.ranges = {
            "Level": self.level,
            "Sill Height": self.sill_height,
            "Orientation": self.orientation,
            "Head Height": self.head_height,
            "Wall Closure": self.wall_closure,
            "Glass Pane Material": self.glass_pane_material,
            "Glass Pane Color": self.glass_pane_color,
            "Sash Material": self.sash_material,
            "Sash Color": self.sash_color,
            "Height": self.height,
            "Width": self.width,
            "Window Inset": self.window_inset
        }
        
    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Wall(Element):
    def __init__(self, name, generation, retrieval):
        super(Wall, self).__init__(name, generation, retrieval)
        self.base_constraint = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.base_offset = [-10000, 10000]
        self.top_constraint = ["Unconnected", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.top_offset = [-10000, 10000]
        self.unconnected_height = [0, 10000]
        self.room_bounding = ["Yes", "No"]
        self.cross_section = ["Vertical", "Slanted", "Tapered"]
        self.angle_from_vertical = [-90, 90]
        self.structural = ["Yes", "No"]
        self.structural_usage = ["Bearing", "Shear", "Structural combined"]
        self.wrapping_at_inserts = ["Do no wrap", "Exterior", "Interior", "Both"]
        self.wrapping_at_ends = ["None", "Exterior", "Interior"]
        self.width = [0, 10000]
        self.function = ["Exterior", "Interior", "Retaining", "Foundation", "Soffit", "Core-shaft"]
        self.material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.ranges = {
            "Base Constraint": self.base_constraint,
            "Base Offset": self.base_offset,
            "Top Constraint": self.top_constraint,
            "Top Offset": self.top_offset,
            "Unconnected Height": self.unconnected_height,
            "Room Bounding": self.room_bounding,
            "Cross-Section": self.cross_section,
            "Angle from Vertical": self.angle_from_vertical,
            "Structural": self.structural,
            "Structural Usage": self.structural_usage,
            "Wrapping at Inserts": self.wrapping_at_inserts,
            "Wrapping at Ends": self.wrapping_at_ends,
            "Width": self.width,
            "Function": self.function,
            "Material": self.material,
            "Color": self.color
        }

    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Stair(Element):
    def __init__(self, name, generation, retrieval):
        super(Stair, self).__init__(name, generation, retrieval)
        self.base_level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.base_offset = [-10000, 10000]
        self.top_level = ["None", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.top_offset = [-10000, 10000]
        self.desired_stair_height = [0, 10000]
        self.desired_number_of_risers = [0, 10000]
        self.actual_tread_depth = [0, 10000]
        self.maximum_riser_height = [0, 10000]
        self.minimum_tread_depth = [0, 10000]
        self.minimum_run_width = [0, 10000]
        self.function = ["Interior", "Exterior"]
        self.right_support = ["None", "Stringer - Closed", "Carriage - Open"]
        self.right_lateral_offset = [0, 10000]
        self.left_support = ["None", "Stringer - Closed", "Carriage - Open"]
        self.left_lateral_offset = [0, 10000]
        self.middle_support = ["Yes", "No"]
        self.middle_support_number = [0, 10000]
        self.ranges = {
            "Base Level": self.base_level,
            "Base Offset": self.base_offset,
            "Top Level": self.top_level,
            "Top Offset": self.top_offset,
            "Desired Stair Height": self.desired_stair_height,
            "Desired Number of Risers": self.desired_number_of_risers,
            "Actual Tread Depth": self.actual_tread_depth,
            "Maximum Riser Height": self.maximum_riser_height,
            "Minimum Tread Depth": self.minimum_tread_depth,
            "Minimum Run Width": self.minimum_run_width,
            "Function": self.function,
            "Right Support": self.right_support,
            "Right Lateral Offset": self.right_lateral_offset,
            "Left Support": self.left_support,
            "Left Lateral Offset": self.left_lateral_offset,
            "Middle Support": self.middle_support,
            "Middle Support Number": self.middle_support_number
        }
    
    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Roof(Element):
    def __init__(self, name, generation, retrieval):
        super(Roof, self).__init__(name, generation, retrieval)
        self.base_level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.room_bounding = ["Yes", "No"]
        self.base_offset_from_level = [-10000, 10000]
        self.rafter_cut = ["Plumb Cut", "Two Cut - Plumb", "Two Cut - Square"]
        self.fascia_depth = [0, 10000]
        self.rafter_or_truss = ["Rafter", "Truss"]
        self.slope = [-360, 360]
        self.material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.thickness = [0, 10000]
        self.ranges = {
            "Base Level": self.base_level,
            "Room Bounding": self.room_bounding,
            "Base Offset From Level": self.base_offset_from_level,
            "Rafter Cut": self.rafter_cut,
            "Fascia Depth": self.fascia_depth,
            "Rafter or Truss": self.rafter_or_truss,
            "Slope": self.slope,
            "Material": self.material,
            "Color": self.color,
            "Thickness": self.thickness
        }

    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Ramp(Element):
    def __init__(self, name, generation, retrieval):
        super(Ramp, self).__init__(name, generation, retrieval)
        self.base_level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.base_offset = [-10000, 10000]
        self.top_level = ["None", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.top_offset = [-10000, 10000]
        self.multistory_top_level = ["None", "Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.width = [0, 10000]
        self.shape = ["Thick", "Solid"]
        self.thickness = [0, 10000]
        self.function = ["Interior", "Exterior"]
        self.material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.maximum_incline_length = [0, 10000]
        self.max_slope = [0, 10000]
        self.ranges = {
            "Base Level": self.base_level,
            "Base Offset": self.base_offset,
            "Top Level": self.top_level,
            "Top Offset": self.top_offset,
            "Multistory Top Level": self.multistory_top_level,
            "Width": self.width,
            "Shape": self.shape,
            "Thickness": self.thickness,
            "Function": self.function,
            "Material": self.material,
            "Color": self.color,
            "Maximum Incline Length": self.maximum_incline_length,
            "Max Slope": self.max_slope
        }

    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Floor(Element):
    def __init__(self, name, generation, retrieval):
        super(Floor, self).__init__(name, generation, retrieval)
        self.level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.height_offset_from_level = [-10000, 10000]
        self.room_bounding = ["Yes", "No"]
        self.structural = ["Yes", "No"]
        self.slope = [-360, 360]
        self.material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.thickness = [0, 10000]
        self.function = ["Interior", "Exterior"]
        self.ranges = {
            "Level": self.level,
            "Height Offset from Level": self.height_offset_from_level,
            "Room Bounding": self.room_bounding,
            "Structural": self.structural,
            "Slope": self.slope,
            "Material": self.material,
            "Color": self.color,
            "Thickness": self.thickness,
            "Function": self.function
        }

    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Door(Element):
    def __init__(self, name, generation, retrieval):
        super(Door, self).__init__(name, generation, retrieval)
        self.level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.sill_height = [-10000, 10000]
        self.orientation = ["Vertical", "Slanted"]
        self.head_height = [-10000, 10000]
        self.wall_closure = ["By host", "Neither", "Interior", "Exterior", "Both"]
        self.function = ["Interior", "Exterior"]
        self.door_material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.door_color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.frame_material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.frame_color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.thickness = [0, 10000]
        self.height = [0, 10000]
        self.trim_projection_exterior = [0, 10000]
        self.trim_projection_interior = [0, 10000]
        self.trim_width = [0, 10000]
        self.width = [0, 10000]
        self.ranges = {
            "Level": self.level,
            "Sill Height": self.sill_height,
            "Orientation": self.orientation,
            "Head Height": self.head_height,
            "Wall Closure": self.wall_closure,
            "Function": self.function,
            "Door Material": self.door_material,
            "Door Color": self.door_color,
            "Frame Material": self.frame_material,
            "Frame Color": self.frame_color,
            "Thickness": self.thickness,
            "Height": self.height,
            "Trim Projection Exterior": self.trim_projection_exterior,
            "Trim Projection Interior": self.trim_projection_interior,
            "Trim Width": self.trim_width,
            "Width": self.width
        }

    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Ceiling(Element):
    def __init__(self, name, generation, retrieval):
        super(Ceiling, self).__init__(name, generation, retrieval)
        self.level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.height_offset_from_level = [-10000, 10000]
        self.room_bounding = ["Yes", "No"]
        self.slope = [-360, 360]
        self.material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.thickness = [0, 10000]
        self.ranges = {
            "Level": self.level,
            "Height Offset from Level": self.height_offset_from_level,
            "Room Bounding": self.room_bounding,
            "Slope": self.slope,
            "Material": self.material,
            "Color": self.color,
            "Thickness": self.thickness
        }
    
    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()

class Column(Element):
    def __init__(self, name, generation, retrieval):
        super(Column, self).__init__(name, generation, retrieval)
        self.base_level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.base_offset = [-10000, 10000]
        self.top_level = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Level 6", "Level 7"]
        self.top_offset = [-10000, 10000]
        self.moves_with_grids = ["Yes", "No"]
        self.room_bounding = ["Yes", "No"]
        self.material = ["Ceramic", "Concrete", "Earth", "Glass", "Masonry", "Metal", "Plastic", "Stone", "Textile", "Wood"]
        self.color = ["Red", "Green", "Blue", "Yellow", "Orange", "Purple", "Pink", "Brown", "Gray", "White", "Black"]
        self.depth = [0, 10000]
        self.offset_base = [0, 10000]
        self.offset_top = [0, 10000]
        self.width = [0, 10000]
        self.ranges = {
            "Base Level": self.base_level,
            "Base Offset": self.base_offset,
            "Top Level": self.top_level,
            "Top Offset": self.top_offset,
            "Moves with Grids": self.moves_with_grids,
            "Room Bounding": self.room_bounding,
            "Material": self.material,
            "Color": self.color,
            "Depth": self.depth,
            "Offset Base": self.offset_base,
            "Offset Top": self.offset_top,
            "Width": self.width
        }

    def generate(self):
        intent = random.choices(self.intents, weights=self.intents_weights)[0]
        if intent == "creation":
            return self.generate_creation()
        elif intent == "deletion":
            return self.generate_deletion()
        elif intent == "retrieval":
            return self.generate_retrieval()
        else: # intent == "modification"
            return self.generate_modification()
