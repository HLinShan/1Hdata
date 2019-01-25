# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 20:19:57 2018

@author: smartdsp
"""

from __future__ import division, print_function
from PIL import Image, ImageFile, ImageOps
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os, sys
reload(sys)
sys.setdefaultencoding( "utf-8" )
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.optim as optim
#from utils import Optimizers
import cv2
#from pooling import WildcatPool2d, ClassWisePool
import random
from skimage.filters import threshold_otsu
#from spaXY_densenet import spaXY_densenet121
#import torch.utils.model_zoo as model_zoo
import xml.etree.cElementTree as et

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#model_urls = {
#    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
#    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
#    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
#    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
#}

def compute_AUCs(gt_np, pred_np):
	
    AUROCs = []
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs
 
def plotCM(classes, y_true, y_pred, savname):
    '''
    classes: a list of class names
    '''
    matrix = confusion_matrix(y_true, y_pred)
    #plot
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center')
    ax.set_xticklabels([' '] + classes, rotation=90)
    ax.set_yticklabels([' '] + classes)
    #save
    plt.savefig(savname)

def add_img_margins(img, h, w):
    '''Add all zero margins to an image
    '''
    enlarged_img = np.zeros((img.shape[0]+h*2, 
                             img.shape[1]+w*2))
    enlarged_img[h:h+img.shape[0], 
                 w:w+img.shape[1]] = img
    return enlarged_img
      
# ====== prepare dataset ======
class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, train_or_valid = "train", 
                 transform=False, angle=[-25, 25], heatmap=False):

        self.train_or_valid = train_or_valid
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:            
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                if label == [0,0]:
                    label = [0]
                elif label == [1,0]:
                    label = [0]
                elif label == [0,1]:
                    label = [1]
                else:
                    label = [1]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)
                
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.angle = angle
        self.heatmap = heatmap
        
        self.label_weight_neg = len(self.labels)/(len(self.labels)-np.sum(self.labels, axis=0))
        self.label_weight_pos = len(self.labels)/(np.sum(self.labels, axis=0))
        
#        self.augm = augm

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        
#        image = cv2.imread(image_name)
#        image = normalize_between(image, 0, 1) * 255
#        image = Image.fromarray(image.astype('uint8')).convert('RGB')
        image = Image.open(image_name).convert('RGB')
        
#        plt.figure("before") # 图像窗口名称
#        plt.imshow(image)
#        plt.axis('off') # 关掉坐标轴为 off
#        plt.title('before') # 图像题目
#        plt.show()
        if self.transform is not None:
            image = self.transform(image)
#        plt.figure("after") # 图像窗口名称
#        plt.imshow(image)
#        plt.axis('off') # 关掉坐标轴为 off
#        plt.title('image') # 图像题目
#        plt.show()
#        asas
        label = self.labels[index]
        label_inverse = np.ones(len(label)) - label
        weight = np.add((label_inverse * self.label_weight_neg),(label * self.label_weight_pos))
        return (image, torch.FloatTensor(label), 
                    torch.from_numpy(weight).type(torch.FloatTensor), image_name)
        
    def __len__(self):
        return len(self.labels)

def normalize_between(img, bottom, top):
    '''
    Normalizes between two numbers: bottom ~ top
    '''
    minimum = np.amin(img, keepdims=True).astype(np.float32)
    maximum = np.amax(img, keepdims=True).astype(np.float32)
    scale_factor = (top - bottom) / (maximum - minimum)
    final_array = (img - minimum) * scale_factor + bottom
    final_array = np.clip(final_array, bottom, top)
    
    return final_array

def print_learning_rate(opt):
    for param_group in opt.param_groups:
        print("Learning rate: %f"%(param_group['lr']))
        

# construct model
#class DenseNet121_wildcat(nn.Module):
#    """Model modified.
#    The architecture of our model is the same as standard DenseNet121
#    except the classifier layer which has an additional sigmoid function.
#    """
#    def __init__(self, num_classes, num_maps=1, kmax=1, kmin=1, alpha=0.7):
#        super(DenseNet121_wildcat, self).__init__()
#        self.densenet121 = models.densenet121(pretrained=True)
#        self.features = self.densenet121.features
#        num_ftrs = self.densenet121.classifier.in_features    
##        model_dict = self.densenet121.state_dict()  
##        pretrain_dict = torch.load(PKL_DIR+'roi_pretrain_best.pkl')
##        pretrain_dict = pretrain_dict['state_dict']
##        pretrained_dict = {k.split('module.')[1]: v for k, v in pretrain_dict.items()}
##        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'classifier'}
##        model_dict.update(pretrained_dict)
#        #fix all resnet50 layer parameters
##        for p in self.densenet121.parameters():
##            p.requires_grad = True
#        self.classifier = nn.Sequential(
#            nn.Conv2d(num_ftrs, num_ftrs//2, kernel_size=3, stride=1, padding=1, bias=True),
#            nn.Conv2d(num_ftrs//2, num_classes*num_maps, kernel_size=3, stride=1, padding=1, bias=True)
#        )
##        self.class_wise = ClassWisePool(num_maps)
##        self.spatial_pool = WildcatPool2d(kmax, kmin, alpha)
#        self.max = nn.AdaptiveMaxPool2d(1)
#
#    def forward(self, x):
#        x = self.features(x)
#        x = F.relu(x, True)
#        xmap = self.classifier(x)
#        xm = F.avg_pool2d(xmap, kernel_size=7, stride=1)
##        xm = self.class_wise(xm)
#        xm = F.sigmoid(xm)
##        x = self.spatial_pool(xm)
#        x = self.max(xm)
##        x = F.softmax(x, dim=1)
#        return xmap, x.view(x.size(0), -1)

# construct model
class DenseNet121_wildcat(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, num_classes):
        super(DenseNet121_wildcat, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        self.features = self.densenet121.features
        num_ftrs = self.densenet121.classifier.in_features 
        self.cls = self.classifier(num_ftrs, num_classes)
    
    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
                nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1, dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1,dilation=1),
                nn.ReLU(True),
                nn.Conv2d(1024, out_planes, kernel_size=1, padding=0)
            )

    def forward(self, x, label=None):
        self.img_erased = x
        x = self.features(x)
        x = F.relu(x,True)
        feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        out = self.cls(feat)
        
#        self.map1 = out
#        localization_map_normed = self.get_atten_map(out, label, True)
        self.attention = out
        
#        logits_1 = F.avg_pool2d(out, 7, 1)
        logits_1 = torch.mean(torch.mean(out, dim=2), dim=2)
#        logits_1 = torch.max(torch.max(out, dim=2)[0], dim=2)[0]
#        logits_1, _ = torch.max(torch.max(out, dim=2)[0], dim=2)
                
        return F.sigmoid(out), F.sigmoid(logits_1)
    
    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed
    
    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
#        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, 0, :,:])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map
    
    def add_heatmap2img(self, img, heatmap):
        # assert np.shape(img)[:3] == np.shape(heatmap)[:3]

        heatmap = heatmap* 255
        color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        img_res = cv2.addWeighted(img.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

        return img_res
    
    def save_erased_img(self, LOG_DIR, img_path, img_batch=None):
        mean_vals = [0.5, 0.5, 0.5]
        std_vals = [0.5, 0.5, 0.5]
        if img_batch is None:
            img_batch = self.img_erased
        if len(img_batch.size()) == 4:
            batch_size = img_batch.size()[0]
            atten_map = F.upsample(self.attention, (1120,448), mode='bilinear')
            
            atten_shape = atten_map.size()        
            pos = torch.ge(F.sigmoid(atten_map), 0.5)
            mask = torch.zeros(atten_shape).cuda()
            mask[pos.data] = 1.0
            mask, _ = torch.max(mask, dim=1, keepdim=True)
            atten_map = F.sigmoid(atten_map).cpu().data.numpy() * mask.cpu().numpy()
            for batch_idx in range(batch_size):
                
                img_dat = img_batch[batch_idx]
                img_dat = img_dat.cpu().data.numpy().transpose((1,2,0))
                img_dat = (img_dat*std_vals + mean_vals)*255
                
                for class_idx in range(atten_shape[1]):
#                    imgname = img_path[batch_idx]
                    nameid = 'atten_'+str(batch_idx)+'_'+str(class_idx)
    
                    mask = atten_map[batch_idx, class_idx, :, :]
    
                    mask = cv2.resize(mask, (448,1120))
                    img_dat = self.add_heatmap2img(img_dat, mask)
                    save_path = os.path.join(LOG_DIR, nameid+'.png')
                    cv2.imwrite(save_path, img_dat)
   
if __name__ == '__main__':
          
    DATA_DIR = './cutimage'
    TRAIN_IMAGE_LIST = './data/full_train_p.txt'
    VALID_IMAGE_LIST = './data/full_test_p.txt'
    HEATMAP_IMAGE_LIST = './data/full_test_p.txt'
    SAVE_DIRS = 'erase/erase_pretrain_mean'
    N_CLASSES = 1
    BATCH_SIZE = 4
    LR = 5e-5
#    correct_pre = 0
    running_loss_val_pre = 100.
    CKPT_NAME = 'DENSENET121_pretrain'#pkl name for saving
    PKL_DIR = 'pkl/'+SAVE_DIRS +'/'
    LOG_DIR = 'logs/' + SAVE_DIRS +'/'
    STEP = 6000
    ste = 1    
#    TRAIN = True
    TRAIN = True
#    Generate_Heatmap = False
    Generate_Heatmap = True
    Pre = False
#    Pre = False
    nmaps = 1
    kmin = 1
    kmax = 1
    OUTPUT_DIR = 'output/' + SAVE_DIRS +'/'
    
    if os.path.isdir(OUTPUT_DIR):
        pass
    else:
        os.mkdir(OUTPUT_DIR) 
    
    CKPT_PATH = PKL_DIR + CKPT_NAME + '_' +str(0)+'.pkl'#pretrain model for loading
        
    # prepare training set
    print('prepare training set...')
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=TRAIN_IMAGE_LIST,
                            train_or_valid="train",
                            transform=transforms.Compose([
                            transforms.Resize([1120,448]),
                            transforms.RandomAffine(25, shear=0.2, scale=(0.8, 1.2)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
#                            transforms.ToPILImage()
                            ]))
    # prepare validation set
    print('prepare validation set...')
    valid_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=VALID_IMAGE_LIST,
                            train_or_valid="valid",
                            transform=transforms.Compose([
                            transforms.Resize([1120,448]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
#                            transforms.ToPILImage()
                            ]))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    
    # prepare validation set
    print('prepare validation set...')
    heatmap_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=HEATMAP_IMAGE_LIST,
                            train_or_valid="valid",
                            transform=transforms.Compose([
                            transforms.Resize([1120,448]),
                            transforms.ToTensor(),
                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                            ]),
                            heatmap=True)
    heatmap_loader = DataLoader(dataset=heatmap_dataset, batch_size=1, shuffle=False)

    # initialize and load the model
    print('initialize and load the model...')

#    model = DenseNet121_wildcat(N_CLASSES, num_maps=nmaps, kmax=kmax, kmin=kmin, alpha=0.7)
    model = DenseNet121_wildcat(N_CLASSES)
#    print(model.state_dict().keys())
#    asas    
    if Pre:
        model_dict = model.state_dict()
        pretrain_dict = torch.load(PKL_DIR+'DENSENET121_pretrain_best' +'.pkl')
        pretrain_dict = pretrain_dict['state_dict']
        pretrained_dict = {k.split('module.')[1]: v for k, v in pretrain_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("=> loaded pretrain checkpoint")
    
    
    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoints = torch.load(CKPT_PATH)
        checkpoint = checkpoints['state_dict']
        ste = checkpoints['step']
        state_dict = {k.split('module.')[1]: v for k, v in checkpoint.items()}
#        print(state_dict.keys())
#        asas
        model.load_state_dict(state_dict)
        print("=> loaded checkpoint: %s"%CKPT_PATH)
    else:
        print("=> no checkpoint found")
        
    if TRAIN:
        if os.path.isdir(PKL_DIR):
            pass
        else:
            os.mkdir(PKL_DIR) 
        if os.path.isdir(LOG_DIR):
            pass
        else:
            os.mkdir(LOG_DIR) 
        writer = SummaryWriter(LOG_DIR)
        # ====== start training =======
        print('start training...')
        cudnn.benchmark = True
        # Get optimizer with correct params.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=LR, weight_decay=1e-5)
#        optimizer = optim.SGD(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7000, gamma=0.1)
        model = torch.nn.DataParallel(model).cuda()
        model.train()
        
        running_loss = 0.0
        
        total_train_length = len(train_dataset)
        total_valid_length = len(valid_dataset)
        perm = np.random.permutation(np.arange(total_train_length))
        cur = 0
#        train_acc = 0.
        for step in range(ste, STEP+1):
            scheduler.step()   
            end = cur + BATCH_SIZE
            p_indexs = perm[cur: end]
            cur = int(cur + BATCH_SIZE)
            if cur > len(train_dataset)-int(BATCH_SIZE):
                cur = 0
                perm = np.random.permutation(np.arange(len(train_dataset)))
                
            augment_img = []
            augment_label = []
            augment_weight = []
            augment_imgname = []
            for p in p_indexs:
                single_img, single_label, single_weight, imgname = train_dataset[p]
                augment_img.append(single_img)
                augment_label.append(single_label)
                augment_weight.append(single_weight)
                augment_imgname.append(imgname)

            inputs_sub = torch.stack(augment_img)
            labels_sub = torch.stack(augment_label)
            weights_sub = torch.stack(augment_weight)
            
            optimizer.zero_grad()
            inputs_sub, labels_sub = Variable(inputs_sub.cuda()), Variable(labels_sub.cuda())
            weights_sub = Variable(weights_sub.cuda())
            weights = np.zeros(N_CLASSES)
            
#                print('forward + backward + optimize...')
            outmap, outp = model(inputs_sub)
#            print(outp.size())
            labels_np = labels_sub.data.cpu().numpy()
#            weights = len(labels_np)/(np.sum(labels_np == 1)+1)
            
            outp_np = outp.data.cpu().numpy()
            
            outmap_np = outmap.data.cpu().numpy()[0,0,:,:]
            
            bce_criterion = nn.BCELoss(size_average=True)
#            criterion = nn.CrossEntropyLoss()
            
            loss = bce_criterion(outp, labels_sub)
#            loss = criterion(outp, labels_sub.squeeze())
            
#            preds = (outp.data > 0.5).type(torch.FloatTensor).cuda()
#            train_acc += torch.sum(preds == labels_sub.data)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
                        
            if step%20 == 0:
                running_loss = running_loss/20
#                train_acc = train_acc / 20 / BATCH_SIZE
                print('[STEP:%d] loss: %.6f' % (step, running_loss))  
#                print('[STEP:%d] loss: %.6f, Acc: %.6f' % (step, running_loss, train_acc))               
                writer.add_scalar('Loss1', running_loss, step)             
#                writer.add_scalar('Acc', train_acc, step)
                running_loss = 0.
#                train_acc = 0.
                model.module.save_erased_img(LOG_DIR, augment_imgname)
                        
            if step%100 == 0:
                model.eval()
                running_loss_val = 0.
                print('Validation Testing......')   
                
                print_learning_rate(optimizer)
                                                
#                lists = ['benign', 'malignant']
#                correct = {'benign': 0., 'malignant': 0.}
                total = {'benign': 0., 'malignant': 0.}
                for p, (inputs_sub, labels_sub, weights_sub, imgname) in enumerate(valid_loader):
        
                    inputs_sub = Variable(inputs_sub.cuda())
                    labels_sub = Variable(labels_sub.cuda())
                    outmap, outp = model(inputs_sub)
#                       print('compute val loss...')
#                    _, predicted = torch.max(F.softmax(outp, dim=1).data, 1)
#                    predicted_np = predicted.cpu().numpy()
#                    predicted_np = (outp.data.cpu().numpy() > 0.5)
#                    _, label = torch.max(labels_sub.data, 1)
#                    target_np = label.cpu().numpy()
#                    target_np = labels_sub.squeeze().data.cpu().numpy()
#                    print(predicted_np, target_np)
#                    correct[lists[int(target_np)]] += int(int(predicted_np) == int(target_np))
#                    total[lists[int(target_np)]] += 1               
#                       print('compute val loss...')
                    loss_val = bce_criterion(outp, labels_sub)
#                    loss_val = criterion(outp, labels_sub.squeeze())
                    running_loss_val += loss_val.data[0]
                    
                running_loss_val = running_loss_val/total_valid_length
                print('[STEP:%d] loss_val: %.6f' % (step, running_loss_val))                
                writer.add_scalar('Loss_val', running_loss_val, step)
                
#                avg_corr = 0.
#                for k in correct:
#                    print('Accuracy of [%s]: %.4f' % (k, correct[k]/total[k]))
#                    avg_corr += correct[k]/total[k]
#                avg_corr /= 2
#                print('Accuracy of the network on test images: %.4f' % (avg_corr))
#                writer.add_scalar('Acc_val', avg_corr, step)
                
                # print statistics
                print('************************************')
#                if avg_corr > correct_pre:
                if running_loss_val < running_loss_val_pre:
                    torch.save({'state_dict': model.state_dict(), 'step': step},PKL_DIR+CKPT_NAME+'_best.pkl')
#                    correct_pre = avg_corr
                    running_loss_val_pre = running_loss_val
                    print('Save best statistics done!')
                torch.save({'state_dict': model.state_dict(), 'step': step},PKL_DIR+CKPT_NAME+'_'+str(step)+'.pkl')
                print('Save [STEP:%d] statistics done!' % (step))
                print('************************************')
                model.train() 
        writer.close()
        print('Finished Training')      
        
    if not Generate_Heatmap:
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        total_valid_length = len(valid_dataset)
        CLASS_NAMES = ['Benign', 'Malignant']
        print(' initialize the ground truth and output tensor...')
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()
        lists = ['benign', 'malignant']
        correct = {'benign': 0., 'malignant': 0.}
        total = {'benign': 0., 'malignant': 0.}
#        lab = []
#        predict = []
        for p, (inputs_sub, labels_sub, weights_sub, imgname) in enumerate(valid_loader):
            if p<2:
                print('the [%d/%d] test'%(p, total_valid_length))
            input_img = Variable(inputs_sub.cuda(), volatile=True)
            labels_sub = Variable(labels_sub.cuda())
#            outmap, outp = model(input_img)
            outmap, outp = model(input_img)
#            _, predicted = torch.max(F.softmax(outp, dim=1).data, 1)
#            predicted_np = predicted.cpu().numpy()
#            predicted_np = (outp.data.cpu().numpy() > 0.5)
#            _, label = torch.max(labels_sub, 1)
#            target_np = label.cpu().numpy()
#            target_np = labels_sub.squeeze().cpu().numpy()
           
#            correct[lists[int(target_np)]] += int(int(predicted_np) == int(target_np))
#            total[lists[int(target_np)]] += 1           
            
            gt = torch.cat((gt, labels_sub.data), 0)
#            lab.append(target_np)
            pred = torch.cat((pred, outp.data), 0)
#            predict.append(predicted_np)
                    
#        avg_corr = 0.
#        for k in correct:
#            print('Accuracy of [%s]: %.4f' % (k, correct[k]/total[k]))
#            avg_corr += correct[k]/total[k]
#        avg_corr /= N_CLASSES
#        print('Accuracy of the network on test images: %.4f' % (avg_corr))
        
#        Plot Confusion Matrix
#        plotCM(CLASS_NAMES, lab, pred.cpu().numpy(), 
#               'Confusion Matrix/'+SAVE_DIRS+'.png')        
#        Compute AUROC            
        print('Compute validation dataset avgAUROC...')    
        gt_npy = gt.cpu().numpy()
        pred_npy = pred.cpu().numpy()
        np.save('./npy/'+SAVE_DIRS+'_gt.npy', gt_npy)
        np.save('./npy/'+SAVE_DIRS+'_pred.npy', pred_npy)
        AUROCs = compute_AUCs(gt_npy, pred_npy)
        AUROC_avg = np.array(AUROCs).mean()
        print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=AUROC_avg))
        for idx in range(N_CLASSES):
            print('The AUROC of {} is {}'.format(CLASS_NAMES[idx], AUROCs[idx]))
    
    #生成热图，定位可视化
    if Generate_Heatmap:
        
        def read_xml_mask(xmlfile, w_, w, h_, h):
            
            scale_w = float(w_)/w
            scale_h = float(h_)/h
            gmask = np.zeros((h_, w_), dtype=np.uint8)
            
            tree=et.parse(xmlfile)
            root=tree.getroot()
            
            for Object in root.findall('object'):
                name=Object.find('name').text.lower()
    #            print(name)                
                bndbox=Object.find('bndbox')
                xmin_scale=int(bndbox.find('xmin').text)*scale_w
                ymin_scale=int(bndbox.find('ymin').text)*scale_h
                xmax_scale=int(bndbox.find('xmax').text)*scale_w
                ymax_scale=int(bndbox.find('ymax').text)*scale_h
                if name == 'calc_malignant' or name == 'mass_malignant':
                    gmask[int(ymin_scale):int(ymax_scale), int(xmin_scale):int(xmax_scale)] = 1
                    
            return gmask
            
        model = torch.nn.DataParallel(model).cuda()
        model.eval()
        total_heatmap_length = len(heatmap_loader)
        CLASS_NAMES = ['Benign', 'Malignant']
        color_map = [(0, 0, 255), (0, 255, 0)]
        font = cv2.FONT_HERSHEY_COMPLEX
        ground = {}
        prediction = {}
        for p, (inputs_sub, labels_sub, weights_sub, img_name) in enumerate(heatmap_loader):
            
            img_name = str(img_name).split('/')[-1].split("'")[0]
            
            target_np = labels_sub.cpu().numpy()
#            print(ill_name)
            if not img_name in ground:
                    ground[img_name] = []
            if int(target_np[0]) == 0:                
                ground[img_name].append(CLASS_NAMES[0])
            if int(target_np[0]) == 1:                
                ground[img_name].append(CLASS_NAMES[1])
                
            if p <2:
                print('the [%d] testimg'%p)
            input_img = Variable(inputs_sub.cuda(), volatile=True)
            probs_tensor, outp = model(input_img)      
            predicted = int(outp.data.cpu().numpy() > 0.226334)
            probs_tensor_np = probs_tensor.data.cpu().numpy()
            
#            print('generate heatmap...')
            
            activate_classes =[0]#predicted#.cpu().numpy()        
            for activate_class in activate_classes:
                activate_class = predicted
                ill_name_pre = CLASS_NAMES[int(activate_class)] 
#                hmask  = probs_tensor_np[0,activate_class,:,:]
                hmask  = probs_tensor_np[0,0,:,:]
                if np.isnan(hmask).any():
                    continue   
                if not img_name in prediction:
                    prediction[img_name] = {}
                if not ill_name_pre in prediction[img_name]:
                    prediction[img_name][ill_name_pre] = {}
                    prediction[img_name][ill_name_pre]['heatmap'] = []  
                prediction[img_name][ill_name_pre]['heatmap'].append(hmask) 
          
        gtall_num = {'Benign': 0, 'Malignant': 0}
        acc = {'Benign': 0, 'Malignant': 0}
        afp = {'Benign': 0, 'Malignant': 0}
        ior = {'Benign': {'count': 0, 'ior': 0}, 'Malignant': {'count': 0, 'ior': 0}}
        for img_name in ground:                
            imgOriginal = cv2.imread(os.path.join(DATA_DIR, img_name), cv2.IMREAD_UNCHANGED)
            h, w = imgOriginal.shape
            imgOriginal = cv2.resize(imgOriginal, (448, 1120))
            x = np.zeros(imgOriginal.shape + (3,), dtype='float32')
            x[:,:,0] = imgOriginal  
            x[:,:,1] = imgOriginal  
            x[:,:,2] = imgOriginal  
            imgOriginal = x
            h_, w_, c = imgOriginal.shape
            gt_num = 0
            positive_num = 0
            for ill_name in ground[img_name]:
                gtall_num[ill_name] += 1    
                xmlname = img_name.split('.')[0]+'.xml'
                gmask = read_xml_mask(os.path.join('./cutxml',xmlname), w_, w, h_, h)
                if np.sum(gmask) == 0:
                    continue
                gmmask = cv2.applyColorMap(gmask, cv2.COLORMAP_RAINBOW) 
                gmmask[np.where(gmask==0)] = 0
                if img_name in prediction:
                    for ill_name_pre in prediction[img_name]:
                        if ill_name_pre == ill_name:
                            hmask = prediction[img_name][ill_name_pre]['heatmap'][0] 
                            hmask[np.where(hmask<0.5)] = 0 
                            hmask = cv2.resize(hmask, (w_, h_))  
                            hmmask = hmask/hmask.max()                  
                            hmmask = cv2.applyColorMap(np.uint8(255*hmmask), cv2.COLORMAP_JET) 
                            
                            img = imgOriginal + gmmask*0.3 + hmmask*0.3
                            outname = os.path.join(OUTPUT_DIR, img_name+'_'+ ill_name_pre +'mask.png')       
                            cv2.imwrite(outname, img)
                                
                            hmask[np.where(hmask!=0)] = 1
                            if np.sum(hmask) == 0:
                                continue
                            iobb = np.sum(hmask*gmask)/np.sum(hmask)   #ior
                            print(iobb)
#                            iobb = np.sum(hmask*gmask)/(np.sum(hmask)+np.sum(gmask)-np.sum(hmask*gmask))  #iou
#                            print(iobb)
                            ior[ill_name]['ior'] += iobb
                            ior[ill_name]['count'] += 1
                            if iobb >= 0.1: acc[ill_name] += 1
                            elif iobb < 0.1: afp[ill_name] += 1                              
                            continue
                                                        
                        else:
                            afp[ill_name_pre] += 1
                    
        ACC = 0.0
        AFP = 0.0
        IOR = 0.0
        for ill_name in gtall_num:
            acc[ill_name] = float(acc[ill_name])/float(gtall_num[ill_name])
            ACC += acc[ill_name]
            afp[ill_name] = float(afp[ill_name])/float(gtall_num[ill_name])
            AFP += afp[ill_name]
            print('The ACC of {} with threshold {} is : {}'.format(ill_name, 0.1, acc[ill_name]))
            print('The AFP of {} with threshold {} is : {}'.format(ill_name, 0.1, afp[ill_name]))  
            if ior[ill_name]['count'] == 0:continue
            ior[ill_name]['avgIoR'] = float(ior[ill_name]['ior'])/float(ior[ill_name]['count'])
            IOR += ior[ill_name]['avgIoR']      
            print('The avgIoR of {} is : {}'.format(ill_name, ior[ill_name]['avgIoR']))  
        print('The average of ACC is : {}'.format(ACC/2))
        print('The average of AFP is : {}'.format(AFP/2))
        print('The average of IOR is : {}'.format(IOR/2))