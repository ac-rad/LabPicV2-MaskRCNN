import os
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms
import json
import shutil
import matplotlib.pyplot as plt


class MedDataset(datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None, transforms=None):
        super(MedDataset, self).__init__(root, transforms, transform, target_transform)
        self.examples = []
        self.root = root
        for sub_folder in os.listdir(self.root):
            for img in os.listdir(os.path.join(self.root, sub_folder)):
                self.examples.append(os.path.join(self.root, sub_folder, img))

    def __getitem__(self, idx):
        data_path = self.examples[idx]
        img = Image.open(data_path).convert("RGB")
        target = {"fname": data_path.split('/')[-1]}
        if self.transforms is not None:
            img, target= self.transforms(img,target)
        return img, target

    def __len__(self):
        return len(self.examples)


class LabPicV2Dataset(datasets.VisionDataset):
    def __init__(self, root, source,  transform=None, target_transform=None, transforms=None, classes=None, subclasses=None, train=True):
        super(LabPicV2Dataset, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.transforms = transforms
        self.source = source
        self.annotations = []
        self.classes = classes
        self.subclass = subclasses
        self.train = train
        self.datapath = self.root + ("/Train" if train else "/Eval")
        print("Creating annotation list for reader this might take a while")
        for AnnDir in os.listdir(self.datapath):
            self.annotations.append(self.datapath+"/"+AnnDir)
        print(self.classes)
        print("Total=" + str(len(self.annotations)))
        print("done making file list")

    def __getitem__(self, idx):
        data_path = self.annotations[idx]
        data = json.load(open(data_path + '/Data.json', 'r'))
        img = Image.open(data_path + "/Image.jpg").convert("RGB")
        num_objs = 0
        labels = []
        sub_class = []
        masks = []
        boxes = []

        def _create_item(data_i, type):
            try:
                labels.append(self.classes[data_i[type][0]])
            except:
                print(data_i)
                print(type)
            sub_label = np.zeros(len(self.subclass) + 1)
            for sub_cls in data_i[type]:
                if sub_cls in self.subclass:
                    sub_label[self.subclass[sub_cls]] = 1
            sub_class.append(sub_label)
            mask = Image.open(data_path + data_i["MaskFilePath"])
            mask = np.array(mask)
            if len(mask.shape) == 3:
                mask = mask[:, :, -1]
            foreGround = mask > 0
            masks.append(foreGround)
            pos = np.where(foreGround)
            try:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            except:
                print(pos)
                print(data_path)

        if "Vessel" in self.source:
            num_objs += len(data["Vessels"])
            for item in data["Vessels"].keys():
                _create_item(data["Vessels"][item], "VesselType_ClassNames")

        if "Material" in self.source:
            num_objs += len(data["MaterialsAndParts"])
            for item in data["MaterialsAndParts"].keys():
                if not (data["MaterialsAndParts"][item]["IsPart"] or data["MaterialsAndParts"][item]["IsOnSurface"] or data["MaterialsAndParts"][item]['IsScattered'] or data["MaterialsAndParts"][item]['IsFullSegmentableMaterialPhase']):
                    _create_item(data["MaterialsAndParts"][item], "MaterialType_ClassNames")

        labels = torch.as_tensor(labels, dtype=torch.int64)
        sub_class = torch.as_tensor(sub_class, dtype=torch.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(list(boxes.size())) < 2:
            print("no box")
            print(data_path)
            return img, {}
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if len(boxes[keep].tolist()) == 0:
            print("no keep box")
            print(boxes)
            print(data_path)
        boxes = boxes[keep]
        labels = labels[keep]
        sub_class = sub_class[keep]
        masks = masks[keep]
        if num_objs == 0:
            area = torch.zeros((num_objs,), dtype=torch.int64)
            print("no object")
            print(data_path)
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "sub_cls": sub_class,
            # "fname": data_path["folder"]
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)

    def convert_semantic(self):
        target_dir = "./Labpic"
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        if not os.path.exists(target_dir + "/image"):
            os.mkdir(target_dir + "/image")
        if not os.path.exists(target_dir + "/mask"):
            os.mkdir(target_dir + "/mask")
        for idx in range(len(self.annotations)):
            data_path = self.annotations[idx]
            data = json.load(open(data_path + '/Data.json', 'r'))
            img = Image.open(data_path + "/Image.jpg").convert("RGB")
            num_objs = 0
            labels = []
            sub_class = []
            masks = []
            boxes = []

            def _create_item(data_i, type):
                try:
                    labels.append(self.classes[data_i[type][0]])
                except:
                    print(data_i)
                    print(type)
                sub_label = np.zeros(len(self.subclass) + 1)
                for sub_cls in data_i[type]:
                    if sub_cls in self.subclass:
                        sub_label[self.subclass[sub_cls]] = 1
                sub_class.append(sub_label)
                mask = Image.open(data_path + data_i["MaskFilePath"])
                mask = np.array(mask)
                if len(mask.shape) == 3:
                    mask = mask[:, :, -1]
                foreGround = mask > 0
                masks.append(foreGround)
                pos = np.where(foreGround)
                try:
                    xmin = np.min(pos[1])
                    xmax = np.max(pos[1])
                    ymin = np.min(pos[0])
                    ymax = np.max(pos[0])
                    boxes.append([xmin, ymin, xmax, ymax])
                except:
                    print(pos)
                    print(data_path)

            if "Vessel" in self.source:
                num_objs += len(data["Vessels"])
                for item in data["Vessels"].keys():
                    _create_item(data["Vessels"][item], "VesselType_ClassNames")

            if "Material" in self.source:
                num_objs += len(data["MaterialsAndParts"])
                for item in data["MaterialsAndParts"].keys():
                    if not (data["MaterialsAndParts"][item]["IsPart"] or data["MaterialsAndParts"][item][
                        "IsOnSurface"] or data["MaterialsAndParts"][item]['IsScattered'] or
                            data["MaterialsAndParts"][item]['IsFullSegmentableMaterialPhase']):
                        _create_item(data["MaterialsAndParts"][item], "MaterialType_ClassNames")

            semantic_mask = np.zeros_like(masks[0])
            for mask in masks:
                semantic_mask += mask
            semantic_mask = semantic_mask > 0
            print(os.path.join(target_dir, f"{idx:09}-transparent-rgb-img.jpg"))
            shutil.copy2(data_path + "/Image.jpg", os.path.join(target_dir, f"image/{idx:09}-transparent-rgb-img.jpg"))
            plt.imsave(os.path.join(target_dir, f"mask/{idx:09}-mask.png"), semantic_mask)


class ChemScapeDataset(datasets.VisionDataset):
    def __init__(self, root, source, readEmpty=False, transform=None, target_transform=None, transforms=None, classes=None,subclasses=None, coco=False):
        super(ChemScapeDataset, self).__init__(root, transforms, transform, target_transform)
        self.root = root
        self.transforms = transforms
        self.source = source
        self.annotations = []
        self.readEmpty = readEmpty
        self.classes = set()
        self.subclass = subclasses
        print("Creating annotation list for reader this might take a while")
        avg_ids = 0
        for AnnDir in os.listdir(self.root):
            for SubDir in self.source:
                path = os.path.join(self.root, AnnDir, SubDir)
                if not os.path.isdir(path):
                    print("No folder:" + path)
                    continue
            if self.readEmpty:
                SubDirs = self.source+ ["EmptyRegions"]
            else:
                SubDirs = self.source
            CatDic = {}
            if coco:
                CatDic["Image"] = os.path.join(self.root, AnnDir, "Image.jpg")
            else:
                CatDic["Image"] = os.path.join(self.root, AnnDir, "Image.png")
            CatDic["folder"] = AnnDir
            CatDic["instances"] = []
            for sdir in SubDirs:
                obj = {}
                InstDir = os.path.join(self.root, AnnDir, sdir)
                if not os.path.isdir(InstDir): continue
                num_instances = 0
                # ------------------------------------------------------------------------------------------------
                for Name in os.listdir(InstDir):
                    num_instances += 1
                    CatString = ""
                    if "CatID_" in Name:
                        CatString = Name[Name.find("CatID_") + 6:Name.find(".png")]
                    ListCat = []
                    if sdir == "EmptyRegions": ListCat = [0]
                    while (len(CatString) > 0):
                        if "_" in CatString:
                            ID = int(CatString[:CatString.find("_")])
                        else:
                            ID = int(CatString)
                            CatString = ""
                        if not ID in ListCat: ListCat.append(ID)
                        CatString = CatString[CatString.find("_") + 1:]
                    obj["Cats"] = ListCat
                    obj["Ann"] = os.path.join(InstDir, Name)
                if num_instances == 0: continue
                CatDic["instances"].append(obj)
                avg_ids += len(obj["Cats"])
                self.classes.update(obj["Cats"])
            if len(CatDic["instances"]) == 0:
                print("No instance" + CatDic["Image"])
                continue
            self.annotations.append(CatDic)
        self.classes = sorted(self.classes)
        self.class_count = 16
        if classes is None:
            self.classes = {self.classes[i]:i+1 for i in range(len(self.classes))}
        else:
            self.classes = classes
        print(self.classes)
        print("Total=" + str(len(self.annotations)))
        print("avg classes per instance=" + str(avg_ids/len(self.annotations)))
        print("done making file list")

    def __getitem__(self, idx):
        data_path = self.annotations[idx]
        img = Image.open(data_path["Image"]).convert("RGB")

        num_objs = len(data_path["instances"])
        labels = []
        sub_class = []
        masks = []
        boxes = []
        for instance in data_path["instances"]:
            if "Cats" not in instance or len(instance["Cats"]) == 0:
                print(data_path["Image"])
                print(instance["Ann"])
            #get the superclass, convert to class index
            labels.append(self.classes[instance["Cats"][0]])
            sub_label = np.zeros(len(self.subclass) + 1)
            sub_class.append(sub_label)
            mask = Image.open(instance["Ann"])
            mask = np.array(mask)
            if len(mask.shape) == 3:
                mask = mask[:, :, -1]
            foreGround = (mask > 0) * (mask < 3)
            masks.append(foreGround)

            pos = np.where(foreGround)
            try:
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                boxes.append([xmin, ymin, xmax, ymax])
            except:
                print(pos)
                print(data_path["Image"])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        sub_class = torch.as_tensor(sub_class, dtype=torch.uint8)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(list(boxes.size())) < 2:
            print(data_path["Image"])
            return img, {}
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if len(boxes[keep].tolist()) == 0 :
            print(boxes)
            print(data_path["Image"])
        boxes = boxes[keep]
        labels = labels[keep]
        sub_class = sub_class[keep]
        masks = masks[keep]
        if num_objs == 0:
            area = torch.zeros((num_objs,), dtype=torch.int64)
            print(data_path["Image"])
        else:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": iscrowd,
            "sub_cls":sub_class,
            #"fname": data_path["folder"]
        }
        if self.transforms is not None:
            img,target = self.transforms(img,target)
        return img, target

    def __len__(self):
        return len(self.annotations)






if __name__=="__main__":
    dataset = LabPicV2Dataset("../../../LabPicData/LabPics2.1/Chemistry", ["Vessel"], classes={"Vessel": 1, "Liquid":2, "Cork": 3, "Solid": 2, "Part":3, "Foam":2}, subclasses={"Syringe": 0, "Pippete":1, "Tube":2, "IVBag": 3, "DripChamber": 4,"IVBottle": 5,"Beaker": 6,"RoundFlask": 7,"Cylinder": 8,"SeparatoryFunnel": 9,"Funnel": 10,"Burete": 11,"ChromatographyColumn": 12,"Condenser": 13,"Bottle": 14,"Jar": 15,"Connector": 16,"Flask": 17,"Cup": 18,"Bowl": 19,"Erlenmeyer": 20,"Vial": 21,"Dish": 22, "HeatingVessel": 23,})
    dataset.convert_semantic()
