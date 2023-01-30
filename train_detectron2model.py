# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances




from detectron2.structures import BoxMode


def get_balloon_dicts(img_dir,labelName):
    json_file = os.path.join(img_dir, labelName+".json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}

        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

def main():
    """
    for d in ["train", "val"]:
        DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("Detectron2-Tutorial/balloon_dataset/balloon/" + d,"via_region_data"))
        MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
    balloon_metadata = MetadataCatalog.get("balloon_train")
    """


    #street_val =detectron2.data.datasets.load_coco_json("data/images/train/labels.json","data/images/train/data","street_train")
    #street_val = detectron2.data.datasets.load_coco_json("D:/unimog data pics/Cam1/mask/labels.json", "D:/unimog data pics/Cam1/org","street_train")
    #street_val = detectron2.data.datasets.load_coco_json("E:/Bankett/labels.json","E:/Bankett", "street_train")
    street_val = detectron2.data.datasets.load_coco_json("E:/Bankett/labels_12_09_2022__20_56.json", "E:/Bankett", "street_train")          #to change


    MetadataCatalog.get("street_train").set(thing_classes=["asphalt"])
    street_metadata = MetadataCatalog.get("street_train")
    DatasetCatalog.register("street_" + "train", lambda d="val":street_val)
    MetadataCatalog.get("street_" + "train").set(thing_classes=["asphalt"])
    #register_coco_instances("street_val",{}, "labels.json", "/data/images/val/")



    dataset_dicts = DatasetCatalog.get("street_train")
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=street_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("image1",out.get_image()[:,:, ::-1]) # change rgb to bgr
        cv2.waitKey(1)


    """
    dataset_dicts = get_balloon_dicts("Detectron2-Tutorial/balloon_dataset/balloon/train","via_region_data")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("image1",out.get_image()[:,:, ::-1]) # change rgb to bgr
        cv2.waitKey(500)

    """

    from detectron2.engine import DefaultTrainer

    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cuda"

    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("street_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512  #128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 49  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if(True):

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        print("start training")
        trainer.train()

    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:

    #cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final .pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold

    predictor = DefaultPredictor(cfg)


    from detectron2.utils.visualizer import ColorMode
    #dataset_dicts = get_balloon_dicts("balloon_dataset/balloon/val")
    dataset_dicts = DatasetCatalog.get("street_train")
    for d in random.sample(dataset_dicts, 1):
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=balloon_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("sepp",out.get_image()[:, :, ::-1])
        cv2.waitKey(200000)

    print("end")

if __name__ == '__main__':
    # freeze_support() here if program needs to be frozen
    main()  # execute this only when run directly, not when imported!