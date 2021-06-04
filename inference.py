import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
from PIL import Image
import os
from Utils import utils
from Reader.InstanceReader.InstanceReaderCoCoStyle import LabPicV2Dataset
from Utils.Visual import ChemDemo
import detection
import os.path as osp
class Maskrcnn(nn.Module):
    def __init__(self,model_location, device, num_classes=3, confidence=0.5, subclass=False):
        super(Maskrcnn, self).__init__()
        self.device = device
        self.confidence = confidence
        self.subclass = subclass
        if self.subclass:
            self.model = detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=False, num_sub_cls=25)
        else:
            self.model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=False)
        checkpoint = torch.load(model_location, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.model.to(device)
        self.demo = ChemDemo(self.model, device=self.device, confidence_threshold=self.confidence)
        self.transform = utils.get_transform(False)

    def compute_prediction(self, image):
        cpu_device = torch.device("cpu")
        image = list(img.to(self.device) for img in image)
        outputs = self.model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        return outputs

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions["scores"]
        keep = torch.nonzero(scores > self.confidence).squeeze(1)
        predictions = {key: predictions[key][keep] for key in predictions.keys()}
        return predictions

    def forward(self, image: list) -> list:
        predictions = self.compute_prediction(image)
        top_predictions = [self.select_top_predictions(p) for p in predictions]
        return top_predictions

    def run_on_one_img(self, img_path, out_dir=None, save_file=None):
        with torch.no_grad():
            img = Image.open(img_path).convert("RGB")
            image = [self.transform(img, None)[0].to(self.device)]
            self.demo.run_on_image(image, target=save_file, outDir=out_dir)


img_path = "/home/sf3202msi/frankapy/test"
model_path = "/home/sf3202msi/PycharmProjects/ChemLab/NEW_V_sub_R50_G.1_A.0025_equal_%x%j/model_160.pth"
device = torch.device('cuda')
mask = Maskrcnn(model_path, device, subclass=True)
total = 10272
with torch.no_grad():
    for subdir, dirs, files in os.walk(img_path):
        imgList = []
        i = 0
        for dirname in dirs:
            if i != 9:
                imgList.append(osp.join(img_path, dirname))
                i += 1
            else:
                imgs = [Image.open(f"{i}/image.jpg").convert("RGB") for i in imgList]
                image = [mask.transform(img, None)[0].to(mask.device) for img in imgs]
                preds = mask(image)
                for i in range(len(preds)):
                    pred = preds[i]
                    masks = np.zeros_like(image[i].cpu())
                    for j in range(len(pred['masks'])):
                        m = pred['masks'][j]
                        masks += (m.numpy() > 0.5) * 5 * (j + 1)
                    masks =  np.einsum('kij->ijk',masks)/255.0
                    plt.imsave(f"{imgList[i]}/labpic_segment.jpg", masks)
                i = 0
                imgList = []