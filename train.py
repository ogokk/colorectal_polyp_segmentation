"""
Deep CNN networks and a new imbalance-aware loss function
for colorectal polyp segmentation
@author: ozangokkan
"""

import gc
import os
import cv2
import time
import warnings
import random
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
import torch 
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from albumentations import (RandomBrightnessContrast ,Flip, HorizontalFlip, VerticalFlip, \
                            ShiftScaleRotate, RGBShift, MotionBlur, OpticalDistortion ,Normalize, Resize, Compose, \
                                GaussNoise,RandomBrightness,RandomCrop,CLAHE)
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchvision import models
# from PIL import Image
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from tqdm import tqdm
from carbontracker.tracker import CarbonTracker
from omni_comprehensive_loss import *
torch.cuda.empty_cache()
warnings.filterwarnings("ignore")

def rle_decode(mask_rle, shape):
    '''
    run-length decoding for masks
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)


class DataAugmentation(Dataset):
    def __init__(self, df, fnames, data_folder, size, mean, std, step):
        self.df = df
        self.root = data_folder
        self.size = size
        self.mean = mean
        self.std = std
        self.step = step
        self.transforms = get_transforms(step, size, mean, std)
        self.gb = self.df.groupby('ImageId')
        self.fnames = fnames

    def __getitem__(self, idx):
        image_id = self.fnames[idx]
        df = self.gb.get_group(image_id)
        annotations = df["EncodedPixels"].tolist()
        image_path = os.path.join(self.root, image_id)
        image = cv2.imread(image_path)
        image = np.asarray(image)
        mask = np.zeros([image.shape[0], image.shape[1]])

        if annotations[0] != '-1':
            for rle in annotations:
                mask += rle_decode(rle, ([image.shape[0], image.shape[1]]))

        mask = (mask >= 1).astype('float32') # for overlap cases
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(step, size, mean, std):
    list_transforms = []
    if step == "train":
        grid_size = np.random.choice([1,2,3,4,5])
        # print(grid_size)
        list_transforms.extend(
            [
                RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.15, brightness_by_max=False, always_apply=False, p=0.5),
                Flip(),
                CLAHE(clip_limit=1, tile_grid_size=(grid_size,grid_size), p=0.5), 
                ShiftScaleRotate(
                    scale_limit=[-0.1,0.5],
                    rotate_limit= int(np.random.choice([0, 10, 45, 90, 135, 180, 270, 300])),
                    p=1.0, # probability of applying the transform
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )
    elif step == "val":
        grid_size = np.random.choice([1,2,3,4,5])
        # print(grid_size)
        list_transforms.extend(
            [
                RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.15, brightness_by_max=False, always_apply=False, p=0.5),
                Flip(),
                CLAHE(clip_limit=1, tile_grid_size=(grid_size,grid_size), p=0.5), 
                ShiftScaleRotate(
                    scale_limit=[-0.1,0.5],
                    rotate_limit= int(np.random.choice([0, 10, 45, 90, 135, 180, 270, 300])),
                    p=1.0, # probability of applying the transform
                    border_mode=cv2.BORDER_CONSTANT
                ),
            ]
        )    
    list_transforms.extend(
        [
            Resize(size, size),
            Normalize(mean=mean, std=std, p=1),
            ToTensor(),
        ]
    )

    list_trfms = Compose(list_transforms)
    return list_trfms

def FoldingAndLoading(fold,total_fold, data_folder,df_path,step,size,mean=None,std=None,batch_size=8,num_workers=0):
    df_all = pd.read_csv(df_path)
    df = df_all.drop_duplicates('ImageId') # remove duplicate annotations
    df["target"] = 1
    kfold = StratifiedKFold(total_fold, shuffle=True, random_state=69)
    indexes_train, indexes_validation= list(kfold.split(df["ImageId"], df["target"]))[fold]
    train_set, val_set = df.iloc[indexes_train], df.iloc[indexes_validation]
    df = train_set if step == "train" else val_set    
    image_names = df['ImageId'].values
    
    image_dataset = DataAugmentation(df_all, image_names, data_folder, size, mean, std, step)
    
    dataloader = DataLoader(image_dataset,batch_size=batch_size,num_workers=num_workers,pin_memory=True,shuffle=True)
    return dataloader


train_rle_path = "C:/Users/ozangokkan/Desktop/PhD/thesis/codes/train-rle.csv"
data_folder    = "C:/Users/ozangokkan/Desktop/PhD/thesis/datasets/images"

# train_rle_path = "C:/Users/ozangokkan/Desktop/PhD/thesis/codes/train-rle-noetis.csv"
# data_folder    = "C:/Users/ozangokkan/Desktop/splitsets/trainvalsets/noetis/images"

# # Dataloader sanity check
# dataloader = FoldingAndLoading(
#     fold=0,
#     total_fold=5,
#     data_folder=data_folder,
#     df_path=train_rle_path,
#     step="train",
#     size=224,
#     mean = (0.485, 0.456, 0.406),
#     std = (0.229, 0.224, 0.225),
#     batch_size=8,
#     num_workers=0,
# )
# batch = next(iter(dataloader)) # get a batch from the dataloader
# images, masks = batch
# rr = iter(dataloader)
# batch = rr.next()
# images, masks = batch



# # plot some random images in the `batch`
# idx = random.choice(range(8))
# plt.figure(1), plt.imshow(images[idx][0], cmap='gray')
# plt.figure(2), plt.imshow(masks[idx][0:], alpha=0.2, cmap='Reds')
# plt.show()
# if len(np.unique(masks[idx][0:])) == 1: # only zeros
#     print('Chosen image has no ground truth mask, rerun the cell')
# #--------------------------------------------------------------------------


  

#For the calculation of IoU and Dice scores
def track_scores(probability, truth, threshold=0.5, reduction='none'):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert(probability.shape == truth.shape)

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        dice = 2 * (p*t).sum(-1)/((p+t).sum(-1))
        
        ious = []
        preds = np.copy(p) 
        labels = np.array(t) # tensor to np
        for pred, label in zip(preds, labels):
            ious.append(np.nanmean(batch_ious(pred, label, classes=[1])))
        iou = np.nanmean(ious)

    return dice, iou

class Observe:
    '''Observe iou and dice scores throughout an epoch'''
    def __init__(self, step, epoch):
        self.base_threshold = 0.5
        self.base_dice_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, iou= track_scores(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice)
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        iou = np.nanmean(self.iou_scores)
        return dice, iou

def batch_ious(pred, label, classes, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]



#cleaning cuda memory
gc.collect()
torch.cuda.empty_cache()

class EncoderDecoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.resnet_layers = list(resnet18.children())

        self.layer0 = nn.Sequential(*self.resnet_layers[:3])
        
        self.layer0_double_ch = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            )
        
        #Basic Block
        self.layer1 = nn.Sequential(*self.resnet_layers[3:5])
        
        self.layer1_double_ch = nn.Sequential(
            nn.Conv2d(64, 64, 1, padding=0),
            nn.ReLU(inplace=True),
            )
                
        #Basic Block
        self.layer2 = nn.Sequential(*self.resnet_layers[5])
        
        self.layer2_double_ch = nn.Sequential(
            nn.Conv2d(128, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            )
        
        #Basic Block
        self.layer3 = nn.Sequential(*self.resnet_layers[6])
        
        self.layer3_double_ch = nn.Sequential(
            nn.Conv2d(256, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            )
        
        #Basic Block [(w-f+2p)/s] +1
        self.layer4 = nn.Sequential(*self.resnet_layers[7])
        
        self.layer4_double_ch = nn.Sequential(
            nn.Conv2d(512, 512, 1, padding=0),
            nn.ReLU(inplace=True),
            )

        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)


        self.convbnrelu3 = nn.Sequential(
            nn.Conv2d(256 + 512, 256, 3, padding=1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            )
        
        self.convbnrelu2 = nn.Sequential(
            nn.Conv2d(128+(256), 256, 3, padding=1),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            )
        
        self.convbnrelu1 = nn.Sequential(
            nn.Conv2d(64+128, 128, 3, padding=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            )
        
        self.convbnrelu0 = nn.Sequential(
            nn.Conv2d(64 + 64, 128, 3, padding=1),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),

            )

        self.conv_original_size0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            )
        
        self.conv_original_size1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            )
        
        self.conv_original_size2 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            )
        
        self.conv_last_one = nn.Sequential(
            nn.Conv2d(64, 32, 1), 
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )
        
        self.conv_last_two = nn.Sequential(
            nn.Conv2d(32, 32, 1),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
            )


        self.conv_last = nn.Conv2d(32, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_double_ch(layer4)
        x = self.upsample(layer4)
        
        layer3 = self.layer3_double_ch(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.convbnrelu3(x)

        x = self.upsample(x)
        layer2 = self.layer2_double_ch(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.convbnrelu2(x)


        x = self.upsample(x)
        layer1 = self.layer1_double_ch(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.convbnrelu1(x)


        x = self.upsample(x)
        layer0 = self.layer0_double_ch(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.convbnrelu0(x)



        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)
        
        x = self.conv_last_one(x)
        x = self.conv_last_two(x)
        
        out = self.conv_last(x)

        return out
    
    
model = EncoderDecoder(n_class=1).cuda()
# model = model.to(device)


# model
#params = list(model.parameters())
#print(len(params))
#print(params[0].size())  # conv1's .weight
#print(params[0])
# from torchinfo import summary as summary_torchinfo
# from torchsummary import summary as summary_torchsummary
# summary_torchsummary(model, (3,448,448))
# summary_torchinfo(model, (3,448,448))

# for name, child in model.named_children():
#     for name2, params in child.named_parameters():
#         print(name, name2)
# Gflops and number of total parameters
# from ptflops import get_model_complexity_info

# with torch.cuda.device(0):
#   gflops, params = get_model_complexity_info(model, (3, 448, 448), as_strings=True,
#                                             print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', gflops))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))


#freezing encoder
# ct = 0
# for child in model.children():
#     ct += 1
#     if ct < 10:
#         for param in child.parameters():
#             param.requires_grad = False
            
            
# m = torch.jit.script(EncoderDecoder(1))            
# torch.jit.save(m, 'scriptmodule.pt')


#Model Training and validation
class StartTraining(object):
    '''Training and validation of our model'''
    def __init__(self, model, total_fold):
        self.fold = np.random.choice(total_fold)
        self.val_no_improve = 0
        self.n_epochs_stop = 10 # number of patience for early stopping criteria
        self.total_fold = total_fold
        self.num_workers = 0
        self.batch_size = 8
        self.accumulation_steps = 32 // self.batch_size
        self.lr = 5e-4
        self.num_epochs = 100
        self.tracker = CarbonTracker(epochs=self.num_epochs, verbose = 2, )
        self.best_loss = 1.0
        self.previous_dice = 0
        self.previous_iou  = 0
        self.steps = ["train", "val"]
        self.device = torch.device("cuda:0")
        self.net = model
        self.criterion = omni_comprehensive_loss()
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            step: FoldingAndLoading(fold=self.fold,total_fold=self.total_fold,data_folder=data_folder,
                df_path=train_rle_path,step=step,size=448,mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),batch_size=8,num_workers=self.num_workers)     
            for step in self.steps
        }

        self.losses = {step: [] for step in self.steps for fold in range(self.total_fold)}
        self.iou_scores = {step: [] for step in self.steps for fold in range(self.total_fold)}
        self.dice_scores = {step: [] for step in self.steps for fold in range(self.total_fold)}
        
        
    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def phase(self, epoch, step):
        scores = Observe(step, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"epoch: {epoch+1} --- step: {step} --- time: {start}")
        self.net.train(step == "train")
        dataloader = self.dataloaders[step]
        current_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(tqdm(dataloader)):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if step == "train":
                loss.backward()
                #Gradient Accumulation
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            current_loss += loss.item()
            outputs = outputs.detach().cpu()
            scores.update(targets, outputs)
        epoch_loss = (current_loss * self.accumulation_steps) / total_batches
        dice, iou = scores.get_metrics() #logging the metrics at the end of an epoch
        print("Loss: %0.4f --- dice: %0.4f --- IoU: %0.4f" % (epoch_loss, dice, iou))
        self.losses[step].append(epoch_loss)
        self.dice_scores[step].append(dice)
        self.iou_scores[step].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, dice, iou

    def start_training(self):
        for epoch in range(self.num_epochs):
            self.tracker.epoch_start()
            self.phase(epoch, "train")
            state = {"epoch": epoch,"best_loss": self.best_loss,# "best_dice": self.previous_dice,# "best_iou" : self.previous_iou,
                     "state_dict": self.net.state_dict(),"optimizer": self.optimizer.state_dict()}
            val_loss, val_dice, val_iou = self.phase(epoch, "val")
            self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                self.val_no_improve = 0
                print("******** Saved best model ********\n")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            else:
                self.val_no_improve +=1
                  # Check early stopping condition
                if self.val_no_improve == self.n_epochs_stop:
                    print('Early stopping!' )
                    early_stop = True
                    break
                else:
                    continue
                if early_stop:
                    print("Stopped")
                    break
            self.tracker.epoch_end()
        print("--------------------------------------------------\n")
        self.tracker.stop()



total_fold = 5
print("\n-----------------------------------------------------")
training_model = StartTraining(model, total_fold)
training_model.start_training()
print("\n-----------------------------------------------------")



# PLOT TRAINING
losses = training_model.losses
dice_scores = training_model.dice_scores # overall dice
iou_scores = training_model.iou_scores

def plot_dice(scores, name):
    plt.figure(figsize=(7,5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.xlabel('Epoch #',fontsize=18); 
    plt.ylabel("Dice scores",fontsize=18)
    plt.legend();
    plt.savefig("dice.png", dpi=600)
    plt.show()


def plot_iou(scores, name):
    plt.figure(figsize=(7,5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.xlabel('Epoch #',fontsize=18); 
    plt.ylabel("IoU scores",fontsize=18)
    plt.legend();
    plt.savefig("iou.png", dpi=600)
    plt.show()
    

def plot_loss(scores, name):
    plt.figure(figsize=(7,5))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.plot(range(len(scores["train"])), scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.xlabel('Epoch #',fontsize=18); 
    plt.ylabel("Loss",fontsize=18)
    plt.legend();
    plt.savefig("loss.png", dpi=600)
    plt.show()
plot_loss(losses, "loss")
plot_dice(dice_scores, "Dice score")
plot_iou(iou_scores, "IoU score")









# #--------------------------------------------------------------------------
# #Test prediction
# #--------------------------------------------------------------------------

# # cleaning cuda memory
# import gc
# gc.collect()
# torch.cuda.empty_cache()

# ## Dataloader   
# class TestDataset(Dataset):
#     def __init__(self, root, df, size, mean, std, tta=4):
#         self.root = root
#         self.size = size
#         self.fnames = (df["ImageId"])
#         self.num_samples = len(self.fnames)
#         self.transform = Compose(
#             [
#                 Normalize(mean=mean, std=std, p=1),
#                 Resize(size, size),
#                 ToTensor(),
#             ]
#         )

#     def __getitem__(self, idx):
#         fname = self.fnames[idx]
#         path = os.path.join(self.root, fname)
#         image = cv2.imread(path)
#         images = self.transform(image=image)["image"]
#         return images

#     def __len__(self):
#         return self.num_samples


# def post_process(probability, threshold, min_region_size,size):
#     mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
#     num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
#     predictions = np.zeros((size, size), np.float32)
#     num = 0
#     for c in range(1, num_component):
# #        p = (component == 1)
#         p = component > 0
#         if p.sum() > min_region_size: # if region of polyp is greater than min_region_size
#             predictions[p] = 1
# #            predictions = component
#             num += 1
#     return predictions, num

# def rle_encode(img):
#     '''
#     Run length encoding for numpy array, 1 - mask, 0 - background
#     '''
#     pixels = img.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return ' '.join(str(x) for x in runs)





# sample_submission_path = "C:/Users/ozangokkan/Desktop/PhD/thesis/codes/test-nokvasir.csv"
# #data_folder    = "C:/Users/ozangokkan/Desktop/PhD/thesis/datasets/images"
# test_data_folder = "C:/Users/ozangokkan/Desktop/splitsets/testingkvasir/images"
# size = 448
# mean = (0.485, 0.456, 0.406) # for 3 channels of an image 
# std = (0.229, 0.224, 0.225)
# num_workers = 0
# batch_size = 8
# best_threshold = 0.5
# min_size = 5
# device = torch.device("cuda:0")
# df = pd.read_csv(sample_submission_path)



# testset = DataLoader(
#     TestDataset(test_data_folder, df, size, mean, std),
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=num_workers,
#     pin_memory=True,
# )
# # --------------------------------------------------------------------------------



# # --------------------------------------------------------------------------------
# model = training_model.net # get the model from model_trainer object
# model.eval()
# state = torch.load("C:/Users/ozangokkan/Desktop/PhD/thesis/codes/model.pth", map_location=lambda storage, loc: storage)
# model.load_state_dict(state["state_dict"])
# # --------------------------------------------------------------------------------






# from PIL import Image
# from torchvision import transforms
# import torchvision
# import torchvision.transforms.functional as F
# import torchvision.transforms as T

# path = "C:/Users/ozangokkan/Desktop/splitsets/testingclinic/images/43.tiff"
# img = cv2.imread(path)
# img = Image.fromarray(img)
# preprocess = T.Compose([
#     T.Resize([448,448]),
#     T.ToTensor(),
#     T.Normalize(
#       mean = [0.485, 0.456, 0.406],
#       std = [0.229, 0.224, 0.225]
#     )
# ])

# x = preprocess(img)
# x.shape

# transformed = torch.unsqueeze(torch.tensor(x), 0)
# print(transformed.shape)
# out = model(transformed)
# out = np.squeeze(out)
# out = out.detach().numpy()
# # plt.imshow(np.int_(out))

# predict_, num_predict_ = post_process(out, best_threshold, min_size,448)


# plt.imshow(np.int_(predict_), alpha=0.2, cmap='Reds')
# plt.show()



# class main_transform():
#     def __init__(self, resize, mean,std):
#         self.transform = transforms.Compose([
#             transforms.Resize([resize,resize]),
#             # transforms.Normalize(mean, std),
#             transforms.ToTensor(),
#             ])
#     def __call__(self, image):
#         return self.transform(image)

# class Predictor():
#     def __init__(self, class_index):
#         self.class_index = class_index
    
#     def predict_max(self, out):
#         max_id = np.argmax(out.detach().numpy())
#         predicted_label_name = self.class_index(str(max_id))
#         return predicted_label_name

# image = Image.open("C:/Users/ozangokkan/Desktop/Polyp.jpg")           
# resize= 448
# mean = [0.485, 0.456, 0.406] # for 3 channels of an image 
# std = [0.229, 0.224, 0.225]

# transformss = main_transform(resize, mean, std)
# transformed =  transformss(image)

# transformed = torch.unsqueeze(torch.tensor(transformed), 0)
# print(transformed.shape)
# out = model(transformed)

# out = out.view(out.shape[2], out.shape[2], out.shape[0])
# print(type(out), out.shape)
# out = out.detach().numpy()

# plt.imshow(out, cmap="gray")
# plt.show()

# plt.imshow(transforms.ToPILImage()(transformed), interpolation="bicubic")
# plt.show()






# # --------------------------------------------------------------------------------

# encoded_pixels = []
# for i, batch in enumerate(tqdm(testset)):
#     preds = torch.sigmoid(model(batch.to(device)))
#     preds = preds.detach().cpu().numpy()[:, 0, :, :] # (batch_size, 1, size, size) -> (batch_size, size, size)
#     for probability in preds:
#         if probability.shape != (size, size):
#             probability = cv2.resize(probability, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
#         predict_, num_predict_ = post_process(probability, best_threshold, min_size,size)
#         if num_predict_ == 0:
#             encoded_pixels.append('-1')
#         else:
#             r = rle_encode(predict_)
#             encoded_pixels.append(r)
# df['EncodedPixels'] = encoded_pixels
# df.to_csv("C:/Users/ozangokkan/Desktop/PhD/thesis/codes/predicted-testkvasir.csv", columns=['ImageId', 'EncodedPixels'], index=False)
