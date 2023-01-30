import glob
import cv2
import torch
import torch.nn as nn
import detectron2
from detectron2.engine import DefaultTrainer
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
import time

import argparse
import os
from typing import Dict, List, Tuple
import torch
from torch import Tensor, nn

import detectron2.data.transforms as T_opt_Model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, detection_utils





from detectron2.structures import BoxMode


dictionary="D:/unimog data pics/Cam1/org/"
dictionary="Detectron2-Tutorial/balloon_dataset/balloon/train/"
dictionary="data/images/val/data/"
dictionary="D:/unimog data pics/seg_mask_keras/Images/"
dictionary="E:/Bankett/Cam1/org/"


# initialize a model with the same architecture as the model which parameters you saved into the .pt/h file

cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"

cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/street_mask_rcnn_R_50_FPN_3x.yaml")
"""
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
"""
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
#torch.save(cfg,"street.yaml")

predictor = DefaultPredictor(cfg)

import detectron2.data.transforms as T

def get_sample_inputs(args):

        # get a sample data
        original_image = detection_utils.read_image(args, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("half").transpose(2, 0, 1)) #changed float32

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs




import torch




from detectron2.modeling import build_model

model_build=build_model(cfg)

from detectron2.checkpoint import DetectionCheckpointer

DetectionCheckpointer(model_build).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS

torch.save(model_build,"model.pt")
#torch.save(predictor,"model.pt")
loadModel=torch.load("model.pt")
loadModel.half()
loadModel.eval()
torch.save(loadModel,"model_half.pt")

#import torch_tensorrt

#torch.onnx.export(loadModel,{"image":torch.randn(3,256,256)},"model.onnx",verbose=False,input_names=["input"],output_names=["output"],export_params=True)


traced_model = torch.load("detectron2/tools/deploy/output/model.ts")
#torch.onnx.export(traced_model,torch.tensor((3,800,1280),dtype=torch.uint8),"model.onnx",opset_version=11,export_params=True,input_names=["input"],output_names=["output"],dynamic_axes={"input":{0:"batch_size"},"output":{0:"batch_size"}})
#traced_model.half()

#shape_of_first_layer = list(loadModel.parameters())
#print(shape_of_first_layer)




MetadataCatalog.get("street_train").set(thing_classes=["asphalt"])
MetadataCatalog.get("street_train").set(thing_colors=[(20,20,20)])
street_metadata = MetadataCatalog.get("street_train")
#street_metadata.thing_colors=[(20,20,20)]

input=get_sample_inputs("detectron2/tools/deploy/resized.jpg")
image = input[0]["image"]
inputs = [{"image": image}]
#flattened_inputs, inputs_schema = flatten_to_tuple(inputs)


from detectron2.utils.visualizer import ColorMode
import PIL
#from torch2trt import torch2trt

for filename in glob.glob(dictionary+'*.bmp'): #media/stefan/Volume/22_06_22/Cam0/org/*.bmp
    im=cv2.imread(filename)
    SI=get_sample_inputs(filename)
    if (False):
        original_image = detection_utils.read_image(filename, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T_opt_Model.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        inputs = {"image": image, "height": height, "width": width}

        # Sample ready
        sample_inputs = [inputs]

        #cv2.imshow("sepp0", im)
        #cv2.imshow("sepp1", original_image)
        #cv2.waitKey(1)

        T_opt_Model = torch.tensor([im[:, :, 2], im[:, :, 1], im[:, :, 0]], dtype=float)

        #traced_model = torch.jit.trace(model_build, example_inputs=sample_inputs)


    device=torch.device("cuda")

    # Import the required libraries
    import torch

    im2=cv2.resize(im,(int(im.shape[1]/1.5),int(im.shape[0]/1.5)))  #resize image to shortest edge
    #cv2.imshow("im",im)
    #cv2.imshow("im2",im2)
    #cv2.waitKey(5000)
    T_opt_Model=torch.tensor([im2[:, :, 0], im2[:, :, 1],  im2[:, :, 2]], dtype=torch.half)
    T_opt_Model_mean=torch.mean(T_opt_Model,(1,2))
    T_opt_Model_mean=T_opt_Model_mean[:,None,None]

    T_traced=torch.tensor([im2[:, :, 0], im2[:, :, 1], im2[:, :, 2]], dtype=torch.half)


    #traced_model = torch.jit.trace(model_build, example_inputs=[[{"image":torch.randn(3,800,1280)}]])
    #T_opt_Model=torch.tensor(T_opt_Model, dtype=torch.half)
    #torch.onnx.export(loadModel,[{"image":torch.randn(3,800,1280)},None], "instance.onnx", input_names=["input"], output_names=["output"],export_params=True)
    #torch.onnx.export(loadModel, [[{"image":T_opt_Model, "height":1200, "width":1920}],{}], "instance.onnx", input_names=["input"], output_names=["output"],export_params=True)


    #model_trt = torch2trt(loadModel, T_opt_Model)
    for x in range(1):
        time0=time.time()
        output_predictor = predictor(im)
        time1 = time.time()
        output_loadModel=loadModel([{"image":T_opt_Model, "height":1200, "width":1920}])
        time2=time.time()
        output_traced=traced_model(T_traced)
        time3 = time.time()
        print("Predictor    : "+str(time1 - time0))
        print("Model opt    : "+str(time2 - time1))
        print("Model traced : "+str(time3 - time2))
    v1 = Visualizer(im[:, :, ::-1],
                   metadata=street_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    v2 = Visualizer(im[:, :, ::-1],
                   metadata=street_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )
    v3 = Visualizer(im[:, :, ::-1],
                   metadata=street_metadata,
                   scale=0.5,
                   instance_mode=ColorMode.IMAGE
                   # remove the colors of unsegmented pixels. This option is only available for segmentation models
                   )

    output=output_loadModel[0]
    tst=torch.tensor(output["instances"].to("cpu")._fields["pred_masks"],dtype=torch.uint8)
    mask=tst.numpy()
    #out_traced_mod=output_traced.to("cpu")*1.5

    out_load = v1.draw_instance_predictions(output["instances"].to("cpu"))
    out_pred=   v2.draw_instance_predictions(output_predictor["instances"].to("cpu"))
    #out_traced_mod=output["instances"].to("cpu")
    #ar=out_traced_mod["_fields"]
    #out_trace = v2.draw_instance_predictions(output_traced[0].to("cpu"))


    #outpu_cpu=output["instances"].to("cpu")
    cv2.imshow("optimized Model", out_load.get_image()[:, :, ::-1])
    cv2.imshow("Predictor",out_pred.get_image()[:, :, ::-1])
    cv2.waitKey(1000)
    print("done")

