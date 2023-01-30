import json

#json_file = "coco_dataset/annotations_trainval2014/annotations/captions_train2014.json"
json_file = "coco_dataset/annotations_trainval2014/annotations/instances_train2014.json"
annotations_dict = {}

with open(json_file) as file:
    annotations_dict = json.load(file)


sample_image = annotations_dict["images"][0]
sample_image_id = sample_image["id"]
sample_image_annotation_index = 33612
sample_image_annotation = annotations_dict["annotations"][sample_image_annotation_index]


# index = 0
# for annotation in annotations_dict["annotations"]:
#     if annotation["image_id"] == sample_image["id"]:
#         sample_image_annotation_index=index
#         break
#         pass
#     index += 1

