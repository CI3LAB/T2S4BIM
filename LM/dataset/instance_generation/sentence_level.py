import os
import json
import re
import random
from utils import Utils
from element_class import Window, Wall, Stair, Roof, Ramp, Floor, Door, Ceiling, Column
from tqdm import tqdm

def rephrase(element, utils, role):
    if role == "user":
        prompt_file = "../prompt/sentence_level_prompt_user.txt"
    elif role == "architect":
        prompt_file = "../prompt/sentence_level_prompt_architect.txt"
    else:
        print("NOT SUCH ROLE!!!")
        exit()
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()
    with open("../information_of_element/{}.txt".format(element), "r", encoding="utf-8") as f:
        information = f.read()
    with open("../data/phrase_{}/{}.json".format(role, element), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_list = []
    for i in tqdm(range(len(data))):
        inputs = prompt.format(element, information, data[i])
        outputs = utils.call_chatgpt(inputs, 0.7)
        output_list.append(outputs)
    with open("../data/sentence_{}/{}.json".format(role, element), "w", encoding="utf-8") as f:
            json.dump(output_list, f, indent=4)

def main():
    role = "architect"
    utils = Utils()
    elements = ["ceiling", "column"]
    for element in elements:
        rephrase(element, utils, role)

if __name__ == "__main__":
    main()
