import os
import json
import re
import random
import openai
import time

class Utils():
    def __init__(self):
        self.option_based_property = ["Wall Closure", "Glass Pane Material", "Glass Pane Color", "Sash Material", "Sash Color", "Level", "Orientation", "Wrapping at Inserts", "Wrapping at Ends",
                                      "Function", "Material", "Color", "Base Constraint", "Top Constraint", "Room Bounding", "Cross-Section", "Structural", "Structural Usage", "Right Support",
                                      "Left Support", "Middle Support", "Base Level", "Top Level", "Rafter Cut", "Rafter or Truss", "Shape", "Multistory Top Level", "Door Material", "Door Color",
                                      "Frame Material", "Frame Color", "Moves with Grids"]
        self.scalar_based_property = ["Height", "Width", "Window Inset", "Sill Height", "Head Height", "Base Offset", "Top Offset", "Unconnected Height", "Angle from Vertical",
                                      "Maximum Riser Height", "Minimum Tread Depth", "Minimum Run Width", "Right Lateral Offset", "Left Lateral Offset", "Middle Support Number",
                                      "Desired Stair Height", "Desired Number of Risers", "Actual Tread Depth", "Thickness", "Base Offset From Level", "Fascia Depth", "Slope",
                                      "Maximum Incline Length", "Max Slope", "Height Offset from Level", "Trim Projection Exterior", "Trim Projection Interior", "Trim Width",
                                      "Depth", "Offset Base", "Offset Top"]
        self.vector_based_property = ["Coordinate"]
        self.no_unit_scalar_based_property = ["Middle Support Number", "Desired Number of Risers", "Max Slope"]
        self.degree_based_property = ["Slope", "Angle from Vertical"]
        self.text_based_property = ["Type Comments", "Comments"]
        self.read_only_property = ["Coordinate", "Length", "Area", "Volume", "Actual Number of Risers", "Actual Riser Height", "Maximum Ridge Height"]

    def mm2m(self, value):
        return value / 1000

    def m2mm(self, value):
        return value * 1000

    def inch2mm(self, inches):
        millimeters = inches * 25.4
        return millimeters

    def mm2inch(self, millimeters):
        inches = millimeters / 25.4
        return inches

    def foot2mm(self, feet):
        millimeters = feet * 304.8
        return millimeters

    def mm2foot(self, millimeters):
        feet = millimeters / 304.8
        return feet

    def degree2radian(self, degree):
        radian = degree * 0.0174533
        return radian

    def radian2degree(self, radian):
        degree = radian / 0.0174533
        return degree

    def count_zeros(self, number):
        number_str = str(number)
        decimal_index = number_str.find('.')

        if decimal_index == -1 or number_str[decimal_index+1:].lstrip('0') == '':
            return 0

        decimal_part = number_str[decimal_index+1:]

        zero_count = 0
        for digit in decimal_part:
            if digit == '0':
                zero_count += 1
            else:
                break

        return zero_count + 1

    def get_value_metric(self, value):
        metrics = ["inch", "mm", "foot", "m"]
        keeps = [1,2,3,0]
        metric = random.choice(metrics)
        k = random.choice(keeps)
        if metric == "inch":
            new_value = self.mm2inch(value)
        elif metric == "foot":
            new_value = self.mm2foot(value)
        elif metric == "mm":
            new_value = value
        else: # metric == "m"
            new_value = self.mm2m(value)
        zeros = self.count_zeros(new_value)
        k = min(k, zeros)
        if k == 0:
            new_value = round(new_value)
        else:
            new_value = round(new_value, k)
        return str(new_value), metric

    def get_value_metric_degree(self, value):
        metrics = ["degree", "radian"]
        keeps = [1,2,3,0]
        metric = random.choice(metrics)
        k = random.choice(keeps)
        if metric == "degree":
            new_value = value
        else: # metric == "radian"
            new_value = self.degree2radian(value)
        zeros = self.count_zeros(new_value)
        k = min(k, zeros)
        if k == 0:
            new_value = round(new_value)
        else:
            new_value = round(new_value, k)
        return str(new_value), metric

    def get_from_to(self, names):
        from_e = random.choice(names)
        new_names = [n for n in names if n != from_e]
        to_e = random.choice(new_names)
        return from_e, to_e

    def get_num(self, begin, end):
        return self.get_value_metric(random.uniform(begin, end))

    def get_num_degree(self, begin, end):
        return self.get_value_metric_degree(random.uniform(begin, end))

    def get_two_nums(self, begin, end):
        num_1, unit_1 = self.get_num(begin, end)
        num_2, unit_2 = self.get_num(begin, end)
        while num_1 == num_2 and unit_1 == unit_2:
            num_2, unit_2 = self.get_num(begin, end)
        return num_1, unit_1, num_2, unit_2

    def get_two_nums_degree(self, begin, end):
        num_1, unit_1 = self.get_num_degree(begin, end)
        num_2, unit_2 = self.get_num_degree(begin, end)
        while num_1 == num_2 and unit_1 == unit_2:
            num_2, unit_2 = self.get_num(begin, end)
        return num_1, unit_1, num_2, unit_2

    def get_times(self):
        nums = ["-1", "-2", "-3", "-4", "-5", "-6", "-7", "-8", "-9", "-10",
                "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        return random.choice(nums), "times"

    def generate_comment(self, property, name):
        if property == "Comments":
            file = "../prompt/comment_prompt.txt"
        else: # property = "Type Comments"
            file = "../prompt/type_comment_prompt.txt"
        with open(file, "r", encoding="utf-8") as f:
            prompt = f.read()
        if property == "Comments":
            prompt = prompt.format(name, name, name)
        else: # property = "Type Comments"
            prompt = prompt.format(name, name, name, name)
        comment = self.call_chatgpt(prompt, 1.0)
        return comment

    def get_words(self, str):
        result = []
        intent_element = re.search(r"intent: (.*?)\n\(element: (.*?)\)\n", str)
        element = intent_element.group(2).strip()
        result.append(element)
        properties = re.findall(r"\(property: [\s\S]+?\)\n", str)
        for property in properties:
            groups = re.findall(r"(.*?): (.*?\n)", property)
            for group in groups:
                if group[1].endswith(")\n"):
                    group_1 = group[1][:-2]
                else:
                    group_1 = group[1]
                result.append(group_1.strip())
        return result

    def parse_string_format(self, str):
        result = {}
        intent_element = re.search(r"intent: (.*?)\n\(element: (.*?)\)\n", str)
        intent = intent_element.group(1).strip()
        element = intent_element.group(2).strip()
        result["intent"] = intent
        result["element"] = element
        properties = re.findall(r"\(property: [\s\S]+?\)\n", str)
        properties_list = []
        for property in properties:
            groups = re.findall(r"(.*?): (.*?\n)", property)
            property_dict = {}
            for group in groups:
                if group[0].startswith("("):
                    group_0 = group[0][1:]
                else:
                    group_0 = group[0]
                if group[1].endswith(")\n"):
                    group_1 = group[1][:-2]
                else:
                    group_1 = group[1]
                property_dict[group_0.strip()] = group_1.strip()
            properties_list.append(property_dict)
        result["property"] = properties_list
        return result

    def parse_json_format(self, dic):
        dic_str = ""
        intent = dic["intent"]
        element = dic["element"]
        dic_str += "(\nintent: {}\n".format(intent)
        if intent == "deletion":
            dic_str += "(element: {})\n)".format(element)
        elif intent == "retrieval":
            dic_str += "(element: {})\n".format(element)
            for property in dic["property"]:
                dic_str += "(property: {})\n".format(property['property'])
            dic_str += ")"
        elif intent == "creation":
            dic_str += "(element: {})\n".format(element)
            for property in dic["property"]:
                dic_str += "(property: {}\n".format(property['property'])
                dic_str += "value: {}\n".format(property['value'])
                if "unit" in property:
                    dic_str += "unit: {})\n".format(property['unit'])
                else:
                    dic_str = dic_str[:-1]
                    dic_str += ")\n"
            dic_str += ")"
        else:
            if intent != "modification":
                print("WRONG INTENT!!!")
                exit()
            dic_str += "(element: {})\n".format(element)
            for property in dic["property"]:
                dic_str += "(property: {}\n".format(property['property'])
                if "source_value" in property:
                    dic_str += "source_value: {}\n".format(property['source_value'])
                    if "source_unit" in property:
                        dic_str += "source_unit: {}\n".format(property['source_unit'])
                dic_str += "target_value: {}\n".format(property['target_value'])
                if "target_unit" in property:
                    dic_str += "target_unit: {})\n".format(property['target_unit'])
                else:
                    dic_str = dic_str[:-1]
                    dic_str += ")\n"
            dic_str += ")"
        return dic_str
            
    def call_chatgpt(self, prompt, temperature):
        openai.api_key = "xxx" # replace with your openai api key
        while True:
            try:
                result = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                ).choices[0].message.content
                break
            except:
                time.sleep(10)
        return result