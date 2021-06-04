# Run  trained net on video to generate prediction and write to another video
# ...............................Imports..................................................................

import cv2
import numpy as np
import torch
import os
import detection
from Utils.Visual import ChemDemo
# import scipy.misc as misc
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--path', default='../LM113Pod2_20200225/W1/20200203', help='dataset')
args = parser.parse_args()
############################################Input parameters###################################################################################
# -------------------------------------Input parameters-----------------------------------------------------------------------
device = torch.device('cuda')  # Use GPU or CPU  for prediction (GPU faster but demend nvidia GPU and CUDA installed else set UseGPU to False)
FreezeBatchNormStatistics = False  # wether to freeze the batch statics on prediction  setting this true or false might change the prediction mostly False work better
OutEnding = ""  # Add This to file name
video_folder = args.path
# -----------------------------------------Location of the pretrain model-----------------------------------------------------------------------------------
Trained_model_path = r"model_400.pth"

##################################Load net###########################################################################################
# ---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
print("predicting subclasses")
model = detection.__dict__['maskrcnn_resnet50_fpn'](num_classes=5, pretrained=False, num_sub_cls=17)
print("loading trained model")
checkpoint = torch.load(Trained_model_path, map_location='cpu')
model.load_state_dict(checkpoint['model'])
demo = ChemDemo(model, device=device, confidence_threshold=0.6)
for hour in os.listdir(video_folder):
    path = os.path.join(video_folder, hour)
    if not os.path.isdir(path): continue
    for minute in os.listdir(os.path.join(video_folder, hour)):
        if len(minute) != 6: continue
        # ---------------------OPEN video-----------------------------------------------------------------------------------------------------
        InputVideo = os.path.join(video_folder, hour, minute)
        print(InputVideo)
        OutVideoMain = InputVideo[
                       :-4] + "_MainClasses.avi"  # Output video that contain vessel filled  liquid and solid
        OutVideoAll = InputVideo[
                      :-4] + "_AllClasses.avi"  # Output video that contain subclasses that have more then 5% of the image
        cap = cv2.VideoCapture(InputVideo)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        MainCatsVideoWriter = None
        AllCatsVideoWriter = None
        # --------------------Create output video---------------------------------------------------------------------------------

        # -----------------------Read Frame one by one-----------------------------------------------------------------------
        # Read until video is completed
        # iii=0
        while (cap.isOpened()):
            # if iii>3: break
            # Capture frame-by-frame
            # ..................Read and resize image...............................................................................

            ret, Im = cap.read()
            if ret == False: break
            # Display the resulting frame

            h, w, d = Im.shape
            r = np.max([h, w])
            if r > 840:  # Image larger then 840X840 are shrinked (this is not essential, but the net results might degrade when using to large images
                fr = 840 / r
                Im = cv2.resize(Im, (int(w * fr), int(h * fr)))
            h, w, d = Im.shape
            if not (type(Im) is np.ndarray): continue
            Imgs = [torch.tensor(np.transpose(Im, (2, 0, 1))) / 255.0]
            # ................................Make Prediction.............................................................................................................
            with torch.autograd.no_grad():
                preditions = demo.predict_on_image(Imgs)
            # ------------------------------------Display main classes on the image----------------------------------------------------------------------------------
            num_obj = len(preditions['labels'])

            MainCatName = ['Liquid', 'Vessel', 'Solid']
            AllCatName = ['Vessel', 'V Label', 'V Cork', 'V Part', 'Ignore', 'Liquid GENERAL',
                          'Liquid Suspension', 'Foam', 'Gel', 'Solid GENERAL', 'Granular', 'Powder',
                          'Solid Bulk', 'Vapor', 'Other', 'Filled']
            results = {name: Im.copy() for name in MainCatName}
            sub_res = {name: Im.copy() for name in AllCatName}
            font = cv2.FONT_HERSHEY_SIMPLEX
            for cls in MainCatName:
                cv2.putText(results[cls], cls, (int(w / 3), int(h / 6)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            for cls in AllCatName:
                cv2.putText(sub_res[cls], cls, (int(w / 3), int(h / 6)), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
            for i in range(num_obj):
                mask = preditions['masks'][i].numpy()
                cls = MainCatName[preditions['labels'][i] - 1]
                res_img = results[cls]
                sub_class = preditions['sub_cls'][i].nonzero().flatten() - 1
                sub_imgs = [sub_res[AllCatName[idx]] for idx in sub_class]
                imgs = [res_img] + sub_imgs
                for img in imgs:
                    img[:, :, 1] = img[:, :, 1] * (1 - mask)
                    img[:, :, 0] = img[:, :, 0] + 255 * mask
            my = 2
            mx = 2
            OutMain = np.zeros([h * my, w * mx, 3], np.uint8)
            OutMain[0:h, 0:w] = results['Vessel']
            OutMain[h: 2 * h, 0:w] = results['Liquid']
            OutMain[0:h, w:2 * w] = results['Solid']
            OutMain[h:2 * h, w:2 * w] = Im
            if MainCatsVideoWriter is None:
                h, w, d = OutMain.shape
                MainCatsVideoWriter = cv2.VideoWriter(OutVideoMain, fourcc, 20.0, (w, h))
            MainCatsVideoWriter.write(OutMain)
            h, w, d = Im.shape
            my = 3
            mx = 3
            OutMain = np.zeros([h * my, w * mx, 3], np.uint8)
            for i in range(9):
                cls = AllCatName[i + 5]
                res = sub_res[cls]
                OutMain[i % 3 * h: (i % 3 + 1) * h, i // 3 * w: (i // 3 + 1) * w] = res
            if AllCatsVideoWriter is None:
                h, w, d = OutMain.shape
                AllCatsVideoWriter = cv2.VideoWriter(OutVideoAll, fourcc, 20.0, (w, h))
            AllCatsVideoWriter.write(OutMain)

        # -----------------------------------------------------------------------------------------------------------------------------
        print("Finished")
        AllCatsVideoWriter.release()
        MainCatsVideoWriter.release()
        cap.release()

