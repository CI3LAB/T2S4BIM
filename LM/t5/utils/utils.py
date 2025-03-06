import os
import json
import re
import random
import openai
import time
from sklearn.metrics import f1_score, accuracy_score

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
        self.no_unit_scalar_based_property = ["Middle Support Number", "Desired Number of Risers", "Max Slope"]
        self.degree_based_property = ["Slope", "Angle from Vertical"]
        self.text_based_property = ["Type Comments", "Comments"]
        self.read_only_property = ["Coordinate", "Length", "Area", "Volume", "Actual Number of Risers", "Actual Riser Height", "Maximum Ridge Height", "Perimeter"]

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
        if intent_element:
            intent = intent_element.group(1).strip()
            element = intent_element.group(2).strip()
        else:
            intent = "not given"
            element = "not given"
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
                try:
                    dic_str += "target_value: {}\n".format(property['target_value'])
                except:
                    print(property)
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
    
    def relative_edit_distance(self, str1, str2):
        m = len(str1)
        n = len(str2)
        dp = [[0] * (n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            dp[i][0] = i
        for j in range(1, n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
        return dp[m][n] / max(m, n)
    
    def remove_non_numeric_chars(self, string):
        pattern = r"[^\d.-]"
        ss = re.sub(pattern, "", string)
        ss = self.remove_extra_symbol(self.remove_in_number(ss))
        if ss == "" or ss == "-" or ss == "." or ss == "-." or ss == ".-":
            return "0"
        return ss
    
    def remove_extra_symbol(self, s):
        if s.count("-") > 1:
            first = s.index("-")
            s = s[:first+1] + s[first+1:].replace("-", "")
        if s.count(".") > 1:
            first = s.index(".")
            s = s[:first+1] + s[first+1:].replace(".", "")
        return s
    
    def remove_in_number(self, number_string):
        if number_string == "":
            return "0"
        first_char = number_string[0]
        if first_char == ".":
            first_char = "0."
        updated_string = first_char + number_string[1:].replace('-', '')
        return updated_string

    def compare_num(self, num_1, unit_1, num_2, unit_2):
        if unit_1 == "times" and unit_2 == "times":
            if num_1 == num_2:
                return True
            else:
                return False
            
        if unit_1 == "inch":
            num_1 = self.inch2mm(num_1)
        elif unit_1 == "foot":
            num_1 = self.foot2mm(num_1)
        elif unit_1 == "m":
            num_1 = self.m2mm(num_1)
        else:
            return False 
        
        if unit_2 == "inch":
            num_2 = self.inch2mm(num_2)
        elif unit_2 == "foot":
            num_2 = self.foot2mm(num_2)
        elif unit_2 == "m":
            num_2 = self.m2mm(num_2)
        else:
            return False
        
        if num_1 - num_2 < 1e-3:
            return True
        else:
            return False
    
    def compare_degree(self, degree_1, unit_1, degree_2, unit_2):
        if unit_1 == "radian":
            degree_1 = self.radian2degree(degree_1)
        if unit_2 == "radian":
            degree_2 = self.radian2degree(degree_2)
        if degree_1 - degree_2 < 1e-3:
            return True
        else:
            return False
    
    def compare_text(self, text_1, text_2):
        rd = self.relative_edit_distance(text_1, text_2)
        if rd < 0.6:
            return True
        else:
            return False
        
    def compare_property(self, property_1, property_2):
        if property_1["property"] != property_2["property"]:
            return False
        if property_1.keys() != property_2.keys():
            return False
        
        if len(property_1.keys()) == 1:
            return True
        
        if property_1["property"] in self.read_only_property:
            return property_1 == property_2
        elif property_1["property"] in self.option_based_property:
            return property_1 == property_2
        elif property_1["property"] in self.no_unit_scalar_based_property:
            return property_1 == property_2
        elif property_1["property"] in self.degree_based_property:
            if "value" in property_1:
                return self.compare_degree(float(self.remove_non_numeric_chars(property_1["value"])), property_1["unit"], float(self.remove_non_numeric_chars(property_2["value"])), property_2["unit"])
            else:
                if "source_value" in property_1:
                    if not self.compare_degree(float(self.remove_non_numeric_chars(property_1["source_value"])), property_1["source_unit"], float(self.remove_non_numeric_chars(property_2["source_value"])), property_2["source_unit"]):
                        return False
                if not self.compare_degree(float(self.remove_non_numeric_chars(property_1["target_value"])), property_1["target_unit"], float(self.remove_non_numeric_chars(property_2["target_value"])), property_2["target_unit"]):
                    return False
            return True
        elif property_1["property"] in self.text_based_property:
            if "value" in property_1:
                return self.compare_text(property_1["value"], property_2["value"])
            else:
                if "source_value" in property_1:
                    if not self.compare_text(property_1["source_value"], property_2["source_value"]):
                        return False
                if not self.compare_text(property_1["target_value"], property_2["target_value"]):
                    return False
            return True
        else:
            if property_1["property"] not in self.scalar_based_property:
                print("WRONG PROPERTY!!!")
                exit()
            if "value" in property_1:
                return self.compare_num(float(self.remove_non_numeric_chars(property_1["value"])), property_1["unit"], float(self.remove_non_numeric_chars(property_2["value"])), property_2["unit"])
            else:
                if "source_value" in property_1:
                    if not self.compare_num(float(self.remove_non_numeric_chars(property_1["source_value"])), property_1["source_unit"], float(self.remove_non_numeric_chars(property_2["source_value"])), property_2["source_unit"]):
                        return False
                if not self.compare_num(float(self.remove_non_numeric_chars(property_1["target_value"])), property_1["target_unit"], float(self.remove_non_numeric_chars(property_2["target_value"])), property_2["target_unit"]):
                    return False
            return True

    def metric_intent(self, label_intents, prediction_intents):
        assert len(label_intents) == len(prediction_intents)
        accuracy = round(accuracy_score(label_intents, prediction_intents) * 100, 2)
        return accuracy
    
    def metric_element(self, label_elements, prediction_elements):
        assert len(label_elements) == len(prediction_elements)
        accuracy = round(accuracy_score(label_elements, prediction_elements) * 100, 2)
        return accuracy
    
    def metric_slot(self, labels, predictions):
        """
        :param y_true: [(index, property), ...]
        :param y_pred: [(index, property), ...]
        :return:
        """
        num_proposed = len(predictions)
        num_gold = len(labels)

        num_correct = 0
        for item in predictions:
            if not item:
                continue
            for pick in labels:
                if item[0] == pick[0] and self.compare_property(item[1], pick[1]):
                    num_correct += 1
                    break

        if num_proposed != 0:
            precision = num_correct / num_proposed
        else:
            precision = 1.0

        if num_gold != 0:
            recall = num_correct / num_gold
        else:
            recall = 1.0

        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        return round(f1 * 100, 2)
    
    def compare_property_list(self, property_list_1, property_list_2):
        if len(property_list_1) != len(property_list_2):
            return False
        for property_1 in property_list_1:
            flag = False
            for property_2 in property_list_2:
                if self.compare_property(property_1, property_2):
                    flag = True
                    break
            if not flag:
                return False
        return True
    
    def compare_instance(self, instance_1, instance_2):
        if instance_1["intent"] != instance_2["intent"]:
            return False
        if instance_1["element"] != instance_2["element"]:
            return False
        if not self.compare_property_list(instance_1["property"], instance_2["property"]):
            return False
        return True
    
    def metric_instance(self, labels, predictions):
        assert len(labels) == len(predictions)
        # all_acc
        # wrong_list_label = []
        # wrong_list_prediction = []
        correct = 0
        for true_label, pred_label in zip(labels, predictions):
            if self.compare_instance(true_label, pred_label):
                correct += 1
            # else:
            #     wrong_list_label.append(true_label)
            #     wrong_list_prediction.append(pred_label)
        all_accuracy = round(correct / len(labels) * 100, 2)
        # with open("wrong_list_label.json", "w") as f:
        #     json.dump(wrong_list_label, f, indent=4)
        # with open("wrong_list_prediction.json", "w") as f:
        #     json.dump(wrong_list_prediction, f, indent=4)

        # accuracy for each category
        intent_labels = ["creation", "deletion", "modification", "retrieval"]
        accuracy_scores = {}

        for label in intent_labels:
            label_indices = []
            for i, instance in enumerate(labels):
                intent = instance["intent"]
                if intent == label:
                    label_indices.append(i)
            label_pred = [predictions[i] for i in label_indices]
            label_true = [labels[i] for i in label_indices]
            
            correct = 0
            for true_label, pred_label in zip(label_true, label_pred):
                if self.compare_instance(true_label, pred_label):
                    correct += 1
            accuracy = round(correct / len(label_true) * 100, 2)
            accuracy_scores[label] = accuracy

            label_slots = [(i, p) for i, d in enumerate(label_true) for p in d["property"]]
            pred_slots = [(i, p) for i, d in enumerate(label_pred) for p in d["property"]]
            slot_f1 = self.metric_slot(label_slots, pred_slots)
            accuracy_scores[label + "_f1"] = slot_f1
        
        # accuracy for each element
        element_labels = ["window", "wall", "stair", "roof", "ramp", "floor", "door", "ceiling", "column"]

        for label in element_labels:
            label_indices = []
            for i, instance in enumerate(labels):
                element = instance["element"]
                if element == label:
                    label_indices.append(i)
            label_pred = [predictions[i] for i in label_indices]
            label_true = [labels[i] for i in label_indices]
            
            correct = 0
            for true_label, pred_label in zip(label_true, label_pred):
                if self.compare_instance(true_label, pred_label):
                    correct += 1
            accuracy = round(correct / len(label_true) * 100, 2)
            accuracy_scores[label] = accuracy

            label_slots = [(i, p) for i, d in enumerate(label_true) for p in d["property"]]
            pred_slots = [(i, p) for i, d in enumerate(label_pred) for p in d["property"]]
            slot_f1 = self.metric_slot(label_slots, pred_slots)
            accuracy_scores[label + "_f1"] = slot_f1
        
        # accuracy for number of property for creation
        creation_number_labels = ["creation_1", "creation_2", "creation_3", "creation_4", "creation_5", "creation_6"]
        
        for label in creation_number_labels:
            label_indices = []
            for i, instance in enumerate(labels):
                if instance['intent'] == "creation" and len(instance['property']) == int(label[-1]):
                    label_indices.append(i)
            label_pred = [predictions[i] for i in label_indices]
            label_true = [labels[i] for i in label_indices]
            
            correct = 0
            for true_label, pred_label in zip(label_true, label_pred):
                if self.compare_instance(true_label, pred_label):
                    correct += 1
            accuracy = round(correct / len(label_true) * 100, 2)
            accuracy_scores[label] = accuracy

            label_slots = [(i, p) for i, d in enumerate(label_true) for p in d["property"]]
            pred_slots = [(i, p) for i, d in enumerate(label_pred) for p in d["property"]]
            slot_f1 = self.metric_slot(label_slots, pred_slots)
            accuracy_scores[label + "_f1"] = slot_f1
        
        # accuracy for number of property for modification
        modification_number_labels = ["modification_1", "modification_2", "modification_3", "modification_4", "modification_5", "modification_6"]

        for label in modification_number_labels:
            label_indices = []
            for i, instance in enumerate(labels):
                if instance['intent'] == "modification" and len(instance['property']) == int(label[-1]):
                    label_indices.append(i)
            label_pred = [predictions[i] for i in label_indices]
            label_true = [labels[i] for i in label_indices]
            
            correct = 0
            for true_label, pred_label in zip(label_true, label_pred):
                if self.compare_instance(true_label, pred_label):
                    correct += 1
            accuracy = round(correct / len(label_true) * 100, 2)
            accuracy_scores[label] = accuracy

            label_slots = [(i, p) for i, d in enumerate(label_true) for p in d["property"]]
            pred_slots = [(i, p) for i, d in enumerate(label_pred) for p in d["property"]]
            slot_f1 = self.metric_slot(label_slots, pred_slots)
            accuracy_scores[label + "_f1"] = slot_f1

        # accuracy for number of property for retrieval
        retrieval_number_labels = ["retrieval_1", "retrieval_2", "retrieval_3", "retrieval_4", "retrieval_5", "retrieval_6"]

        for label in retrieval_number_labels:
            label_indices = []
            for i, instance in enumerate(labels):
                if instance['intent'] == "retrieval" and len(instance['property']) == int(label[-1]):
                    label_indices.append(i)
            label_pred = [predictions[i] for i in label_indices]
            label_true = [labels[i] for i in label_indices]
            
            correct = 0
            for true_label, pred_label in zip(label_true, label_pred):
                if self.compare_instance(true_label, pred_label):
                    correct += 1
            accuracy = round(correct / len(label_true) * 100, 2)
            accuracy_scores[label] = accuracy

            label_slots = [(i, p) for i, d in enumerate(label_true) for p in d["property"]]
            pred_slots = [(i, p) for i, d in enumerate(label_pred) for p in d["property"]]
            slot_f1 = self.metric_slot(label_slots, pred_slots)
            accuracy_scores[label + "_f1"] = slot_f1
  
        return all_accuracy, accuracy_scores


    def metric(self, label_texts, predictions):
        assert len(label_texts) == len(predictions)
        label_dicts = [self.parse_string_format(label_text) for label_text in label_texts]
        prediction_dicts = [self.parse_string_format(prediction) for prediction in predictions]
        
        gold_intents = [d["intent"] for d in label_dicts]
        pred_intents = [d["intent"] for d in prediction_dicts]
        intent_acc = self.metric_intent(gold_intents, pred_intents)

        gold_elements = [d["element"] for d in label_dicts]
        pred_elements = [d["element"] for d in prediction_dicts]
        element_acc = self.metric_element(gold_elements, pred_elements)

        label_slots = [(i, p) for i, d in enumerate(label_dicts) for p in d["property"]]
        pred_slots = [(i, p) for i, d in enumerate(prediction_dicts) for p in d["property"]]
        slot_f1 = self.metric_slot(label_slots, pred_slots)

        # property-f1 for each type
        label_slots = [(i, p) for i, d in enumerate(label_dicts) for p in d["property"] if p["property"] in self.option_based_property]
        pred_slots = [(i, p) for i, d in enumerate(prediction_dicts) for p in d["property"] if p["property"] in self.option_based_property]
        option_based_property_f1 = self.metric_slot(label_slots, pred_slots)

        label_slots = [(i, p) for i, d in enumerate(label_dicts) for p in d["property"] if p["property"] in self.scalar_based_property]
        pred_slots = [(i, p) for i, d in enumerate(prediction_dicts) for p in d["property"] if p["property"] in self.scalar_based_property]
        scalar_based_property_f1 = self.metric_slot(label_slots, pred_slots)

        label_slots = [(i, p) for i, d in enumerate(label_dicts) for p in d["property"] if p["property"] in self.text_based_property]
        pred_slots = [(i, p) for i, d in enumerate(prediction_dicts) for p in d["property"] if p["property"] in self.text_based_property]
        text_based_property_f1 = self.metric_slot(label_slots, pred_slots)

        label_slots = [(i, p) for i, d in enumerate(label_dicts) for p in d["property"] if p["property"] in self.read_only_property]
        pred_slots = [(i, p) for i, d in enumerate(prediction_dicts) for p in d["property"] if p["property"] in self.read_only_property]
        read_only_property_f1 = self.metric_slot(label_slots, pred_slots)

        all_acc, acc_scores = self.metric_instance(label_dicts, prediction_dicts)

        acc_scores["option_based_f1"] = option_based_property_f1
        acc_scores["scalar_based_f1"] = scalar_based_property_f1
        acc_scores["text_based_f1"] = text_based_property_f1
        acc_scores["read_only_f1"] = read_only_property_f1

        return intent_acc, element_acc, slot_f1, all_acc, acc_scores
