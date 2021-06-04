# ChemLab

Env install please refer to env.yaml

To get the MaskRcnn pretrained on our LabPic V2 dataset, just clone the repo with git lfs. (model_160.pth)

Testing

```
python train.py --test-only --resume model.pth --dataset Vessel Material

```

Training

```
python train.py --resume model.pth --dataset Vessel Material
```

To use it for inference

```
from inference import Maskrcnn
model_path = "/home/sf3202msi/PycharmProjects/ChemLab/NEW_V_sub_R50_G.1_A.0025_equal_%x%j/model_160.pth"
device = torch.device('cuda')
mask = Maskrcnn(model_path, device, subclass=True)
imgs = [Image.open(f"{i}/image.jpg").convert("RGB") for i in imgList]
image = [mask.transform(img, None)[0].to(mask.device) for img in imgs]
preds = mask(image)
```

