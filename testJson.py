import json

json_file = "yolov7/datasets/street/annotations/labels_val.json"
annotations_dict = {}

with open(json_file) as file:
    annotations_dict = json.load(file)

print("done")
