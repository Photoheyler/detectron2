import fiftyone as fo
import fiftyone.zoo as foz

#dataset = foz.load_zoo_dataset("coco-2017",max_samples=2,label_types=["detections","segmentations"])

dataset_dir='C:/Users/Photoheyler/Desktop/YoloV7Instance/data/images/val'

dataset_type= fo.types.COCODetectionDataset
name="train"


dataset=fo.Dataset.from_dir(dataset_dir=dataset_dir,dataset_type=dataset_type,name="train")

session = fo.launch_app(dataset,
                        desktop=True)
session.wait()