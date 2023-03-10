'''
Create a Swin Transformer model as defined in the paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows",
then connect the backbone to a instance segmentation head using Mask R-CNN.
'''
from torchvision.models import swin_b
from torch.nn import Sequential
import torchvision
import detection
from mmdet.apis import init_detector
from torch import cuda
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

def swin_mask_rcnn(num_classes, pretrained=False):
    # create a Swin Transformer backbone
    # Pop the last three layers of the Swin Transformer backbone to remove the flattening and linear head
    backbone = swin_b(weights='Swin_B_Weights.DEFAULT')
    children = list(backbone.children())
    for i in range(3):
        children.pop()
    backbone = Sequential(*children)
    # create a Mask R-CNN model using the Swin Transformer backbone
    model = torchvision.models.detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=pretrained)
    fpn = FeaturePyramidNetwork(
        in_channels_list=[256, 512, 1024, 2048],
        out_channels=256,
        extra_blocks=LastLevelMaxPool()
    )
    # fpn = model.backbone.fpn
    # model = detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=pretrained, num_sub_cls=25)
    model.backbone = backbone
    # Add the fpn to the Swin Transformer backbone
    model.backbone.fpn = fpn
    # config_file = '/home/alexliu/Dev/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco.py'
    # model = init_detector(config_file)
    # Freeze the weights of the backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    return model

if __name__ == '__main__':
    model = swin_mask_rcnn(3, pretrained=False)
    print(model)