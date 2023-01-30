import json
import glob
import os
import cv2
from datetime import datetime

json_file = "labels.json"
annotations_dict = {}

MaskFolder='D:/unimog data pics/Cam1/mask/'
MaskFolder='C:/Users/Photoheyler/Desktop/YoloV7Instance/data/mask/train/'


rootDirectory="E:/Bankett/"             # To change

FolderList=os.listdir(rootDirectory)
FolderList=["08_12_2022 14_07_38 Lidx0"]
with open(json_file) as file:
    annotations_dict = json.load(file)

annotations_dict["info"] = {"date_created":"17_10_22","description":"automatic generated"}
images=[]
i=0
annotations=[]

for Foldername in FolderList:
    MaskFolder=rootDirectory+Foldername+"/mask/"
    for filename in glob.glob(MaskFolder +'*.bmp'):
        head, tail = os.path.split(filename)
        im = cv2.imread(filename)
        rows, cols, channels = im.shape
        images.append({"id":i,"width" : cols,"height":rows,"file_name":Foldername+"/org/"+tail})

        print(filename)




        head, tail = os.path.split(filename)
        im=cv2.imread(filename)
        rows,cols,channels=im.shape
        xValTop=0
        firstpx=False
        for c in range(cols):
            if(im[0,c,0]==1 and firstpx==False):
                xValTop=float(c)
                firstpx=True
        firstpx=False
        for c in range(cols):
            if(im[rows-1,c,0]==1 and firstpx==False):
                xValBot=float(c)
                firstpx=True
        widestx=0
        if(xValTop>xValBot):
            widestx=xValTop
        else:
            widestx=xValBot

        annotations.append({"id":i,"image_id":i,"bbox":[0.0,0.0,widestx,rows-1.0],"area":0.0,"iscrowd":0,"category_id":0,"segmentation":[[0.0,0.0,xValTop,0.0,xValBot,rows-1.0,0.0,rows-1.0,0.0,0.0]]})
        print(i)
        i += 1
    annotations_dict["images"] += images
    print(images)

annotations_dict["annotations"]=annotations
annotations_dict["categories"]=[{"id":0,"name":"asphalt"}]
annotations_dict["licenses"]={}

json_store_file = "labels_exp.json"

now = datetime.now() # current date and time
with open(rootDirectory+"labels"+now.strftime("_%m_%d_%Y__%H_%M")+".json", "w") as file:
    json.dump(annotations_dict, file)

customizedAnnotatons={}
with open(json_file) as file:
    customizedAnnotatons = json.load(file)

print("end")