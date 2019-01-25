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

def cut_img_margins(img, h, w):
    '''Add all zero margins to an image
    '''
    cut_img = np.zeros((img.shape[0]-h*2, 
                             img.shape[1]-w*2))
    cut_img = img[h:img.shape[0]-h, 
                       w:img.shape[1]-w]
    return cut_img
            
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
                image_name= items[0].split('.')[0]+'.'+items[0].split('.')[1]+'.npy'
                label = items[1]
                label = [int(i) for i in label]
                if label == [0]:
                    label = [0]
                elif label == [1]:
                    label = [0]
                elif label == [2]:
                    label = [1]
                elif label == [3]:
                    label = [1]
                else:
                    label = [0]
#                label = np.eye(N_CLASSES, dtype=int)[label[0]]#onehot
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
        
        self.randomcrop = transforms.RandomCrop([224,224])
        self.centercrop = transforms.CenterCrop([224,224])

    def __getitem__(self, index):
        """
        Args:
            index: the index of item 
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
#        image = cv2.imread(image_name, cv2.IMREAD_UNCHANGED)
#        image = image / 65535. *255.
#        h, w = image.shape
#        plt.figure("roi") # 图像窗口名称
#        plt.imshow(image)
#        plt.axis('off') # 关掉坐标轴为 off
#        plt.title('roi') # 图像题目
#        plt.show()
#        if h < 224:
#            image = add_img_margins(image, int(np.ceil((224-h)/2)), 0)
#        else:
#            image = cut_img_margins(image, (h-224)//2, 0)
#        if w < 224:
#            image = add_img_margins(image, 0, int(np.ceil((224-w)/2)))
#        else:
#            image = cut_img_margins(image, 0, (w-224)//2)            
            
#        image = Image.fromarray(image.astype('uint8')).convert('RGB')
#        if h > 224 or w > 224:
#            if self.train_or_valid == 'train':
#                image = self.randomcrop(image)
#            else:
#                image = self.centercrop(image)
        
#        image = Image.open(image_name).convert('RGB')
        image = np.load(image_name) * 255
        image = Image.fromarray(image.astype('uint8')).convert('RGB')
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
#        plt.title('after') # 图像题目
#        plt.show()
#        asas
        label = self.labels[index]
        label_inverse = np.ones(len(label)) - label
        weight = np.add((label_inverse * self.label_weight_neg),(label * self.label_weight_pos))
        return (image, torch.FloatTensor(label), 
                torch.from_numpy(weight).type(torch.FloatTensor), image_name)
        
    def __len__(self):
        return len(self.labels)

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
#        model_dict = self.densenet121.state_dict()  
#        pretrain_dict = torch.load(PKL_DIR+'ddsm_pretrain.pkl')
#        pretrain_dict = pretrain_dict['state_dict']
#        pretrained_dict = {k.split('module.')[1]: v for k, v in pretrain_dict.items()}
#        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k != 'classifier'}
#        model_dict.update(pretrained_dict)
#        #fix all resnet50 layer parameters
##        for p in self.densenet121.parameters():
##            p.requires_grad = True
#        self.classifier = nn.Linear(num_ftrs, num_classes)
#
#    def forward(self, x):
#        x = self.features(x)
#        x = F.relu(x, True)
#        x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), stride=1).view(x.size(0), -1)
#        x = self.classifier(x)
##        x = F.softmax(x, dim=1)
#        return x
    
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
        # print(x.shape) #12*7*7*1024
        feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        # print("shape",feat.shape) 12*7*7*1024
        out = self.cls(feat)
        # print(out.shape) #12 7*7*1
        
        self.map1 = out
        #　make heatmap 7＊7
        localization_map_normed = self.get_atten_map(out, label, True)
        self.attention = localization_map_normed
        # print(self.attention.shape)#１２＊７＊７　
        
        logits_1 = torch.mean(torch.mean(out, dim=2), dim=2)
        # print(logits_1.shape)
                
        return F.sigmoid(logits_1)
    
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
            atten_map = F.upsample(self.attention.unsqueeze(dim=1), (224,224), mode='bilinear')
            for batch_idx in range(batch_size):
                imgname = img_path[batch_idx]
                nameid = imgname.strip().split('/')[-1].strip().split('.')[0]

                # atten_map = F.upsample(self.attention.unsqueeze(dim=1), (321,321), mode='bilinear')
#                atten_map = F.upsample(self.attention.unsqueeze(dim=1), (224,224), mode='bilinear')
#                atten_map = F.upsample(self.attention, (224,224), mode='bilinear')
                # mask = F.sigmoid(20*(atten_map-0.5))
                mask = atten_map[batch_idx, :, :]
                mask = mask.squeeze().cpu().data.numpy()
#                print(mask.shape)

                img_dat = img_batch[batch_idx]
                img_dat = img_dat.cpu().data.numpy().transpose((1,2,0))
                img_dat = (img_dat*std_vals + mean_vals)*255

                mask = cv2.resize(mask, (224,224))
                img_dat = self.add_heatmap2img(img_dat, mask)
                save_path = os.path.join(LOG_DIR, nameid+'.png')
                cv2.imwrite(save_path, img_dat)
    
if __name__ == '__main__':
          
    DATA_DIR = './roiimage_'
    TRAIN_IMAGE_LIST = './data/full_roi_train_p.txt'
    VALID_IMAGE_LIST = './data/full_roi_test_p.txt'
    HEATMAP_IMAGE_LIST = './data/full_roi_test_p.txt'
    SAVE_DIRS = 'Densenet121_pretrain_MIL_224_aug_lr5e-5_class2_1H_roi_p_mean_npy'
    N_CLASSES = 1
    BATCH_SIZE = 12
    LR = 5e-5
    correct_pre = 0
    running_loss_val_pre = 100.
    CKPT_NAME = 'DENSENET121_pretrain'#pkl name for saving
    PKL_DIR = 'pkl/'+SAVE_DIRS +'/'
    LOG_DIR = 'logs/' + SAVE_DIRS +'/'
    STEP = 4000
    ste = 1    
    TRAIN = True
#    TRAIN = False
    Generate_Heatmap = False
    Pre = False
#    nmaps = 1
#    kmin = 1
#    kmax = 1
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
                            transforms.Resize([224,224]),
#                            transforms.RandomCrop([224,224]),
                            transforms.RandomAffine(360, shear=0.2, scale=(0.8, 1.2)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
#                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
#                            transforms.ToPILImage()
                            ]))
    # prepare validation set
    print('prepare validation set...')
    valid_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=VALID_IMAGE_LIST,
                            train_or_valid="valid",
                            transform=transforms.Compose([
                            transforms.Resize([224,224]),
                            transforms.ToTensor(),
#                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
#                            transforms.ToPILImage()
                            ]))
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    
    # prepare validation set
    print('prepare validation set...')
    heatmap_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                            image_list_file=HEATMAP_IMAGE_LIST,
                            train_or_valid="valid",
                            transform=transforms.Compose([
                            transforms.Resize([224,224]),
                            transforms.ToTensor(),
#                            transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
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
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.1)
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
            print("training",inputs_sub.shape,labels_sub.shape)
            outp = model(inputs_sub, labels_sub.squeeze())
#            print(outp.size())
#            labels_np = labels_sub.data.cpu().numpy()
#            weights = len(labels_np)/(np.sum(labels_np == 1)+1)
            
            outp_np = outp.data.cpu().numpy()
            
#            outmap_np = outmap.data.cpu().numpy()[0,0,:,:]
            
            bce_criterion = nn.BCELoss(size_average=True)
#            criterion = nn.CrossEntropyLoss()
            
            loss = bce_criterion(outp, labels_sub)
#            loss = criterion(outp, labels_sub.squeeze())
#            _, predicted = torch.max(F.softmax(outp, dim=1).data, 1)
#            print(predicted[0] == labels_sub.data[0])
#            for i in range(outp_np.shape[0]):
#                train_acc += torch.sum(predicted[i] == labels_sub.data[i])
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
                        
            if step%20 == 0: #　two_classification
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
                                
                # lists = ['cb', 'mb', 'cm', 'mm', 'nor']
                # correct = {'cb': 0., 'mb': 0., 'cm': 0., 'mm': 0., 'nor':0.}
                # total = {'cb': 0., 'mb': 0., 'cm': 0., 'mm': 0., 'nor':0., }
                for p, (inputs_sub, labels_sub, weights_sub, imgname) in enumerate(valid_loader):
        
                    inputs_sub = Variable(inputs_sub.cuda())
                    labels_sub = Variable(labels_sub.cuda())
                    print("test",inputs_sub.shape,labels_sub.shape)
                    outp = model(inputs_sub, labels_sub.squeeze())

#                       print('compute val loss...')
#                    _, predicted = torch.max(F.softmax(outp, dim=1).data, 1)
#                    predicted_np = predicted.cpu().numpy()
#                    _, label = torch.max(labels_sub.data, 1)
#                    target_np = label.cpu().numpy()
                    target_np = labels_sub.squeeze().data.cpu().numpy()
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
#                avg_corr /= N_CLASSES
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
        CLASS_NAMES = ['calcification_Benign', 'mass_Benign', 
                       'calcification_Malignant', 'mass_Malignant', 'normal']
        print(' initialize the ground truth and output tensor...')
        gt = torch.FloatTensor()
        gt = gt.cuda()
        pred = torch.FloatTensor()
        pred = pred.cuda()
#        lists = ['cb', 'mb', 'cm', 'mm', 'nor']
#        correct = {'cb': 0., 'mb': 0., 'cm': 0., 'mm': 0., 'nor':0.}
#        total = {'cb': 0., 'mb': 0., 'cm': 0., 'mm': 0., 'nor':0., }
        for p, (inputs_sub, labels_sub, weights_sub, imgname) in enumerate(valid_loader):
            
            if p<2:
                print('the [%d/%d] test'%(p, total_valid_length))
                
            input_img = Variable(inputs_sub.cuda(), volatile=True)
            labels_sub = Variable(labels_sub.cuda())
            
            outp = model(input_img, labels_sub.squeeze())
            
#            _, predicted = torch.max(F.softmax(outp, dim=1).data, 1)
#            predicted_np = predicted.cpu().numpy()
#            target_np = labels_sub.squeeze().data.cpu().numpy()
           
#            correct[lists[int(target_np)]] += int(int(predicted_np) == int(target_np))
#            total[lists[int(target_np)]] += 1           
            
            gt = torch.cat((gt, labels_sub.data), 0)
            pred = torch.cat((pred, outp.data), 0)
                    
#        avg_corr = 0.
#        for k in correct:
#            print('Accuracy of [%s]: %.4f' % (k, correct[k]/total[k]))
#            avg_corr += correct[k]/total[k]
#        avg_corr /= N_CLASSES
#        print('Accuracy of the network on test images: %.4f' % (avg_corr))
        
#        Plot Confusion Matrix
#        gtmax = gt.squeeze().cpu().numpy()
#        _, predmax = torch.max(pred, 1)
#        plotCM(CLASS_NAMES, gtmax, predmax.cpu().numpy(), 
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
#    
    #生成热图，定位可视化
    if Generate_Heatmap:
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
            predicted = int(outp.data.cpu().numpy() > 0.5)
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
            imgOriginal = cv2.imread(os.path.join(DATA_DIR, img_name.split('_')[-1].split('.')[0]+'.jpg'), cv2.IMREAD_UNCHANGED)
            imgOriginal = cut(imgOriginal)
            imgOriginal = cv2.resize(imgOriginal, (448, 576))
            x = np.zeros(imgOriginal.shape + (3,), dtype='float32')
            x[:,:,0] = imgOriginal  
            x[:,:,1] = imgOriginal  
            x[:,:,2] = imgOriginal  
            imgOriginal = x
            w_ori = np.shape(imgOriginal)[0]
            h_ori = np.shape(imgOriginal)[1]
            gt_num = 0
            positive_num = 0
            for ill_name in ground[img_name]:
                ids = img_name.split('_')[-1]
                if os.path.exists(os.path.join(DATA_DIR, ids+'_mask.png')):
                    gtall_num[ill_name] += 1    
                if img_name in prediction:
                    for ill_name_pre in prediction[img_name]:
                        if ill_name_pre == ill_name:
                            hmask = prediction[img_name][ill_name_pre]['heatmap'][0] 
                            hmask[np.where(hmask<0.5*hmask.max())] = 0 
                            hmask = cv2.resize(hmask, (h_ori, w_ori))  
                            hmmask = hmask/hmask.max()                  
                            hmmask = cv2.applyColorMap(np.uint8(255*hmmask), cv2.COLORMAP_JET) 
                            if os.path.exists(os.path.join(DATA_DIR, ids+'_mask.png')):
                                gmask = cv2.imread(os.path.join(DATA_DIR, ids+'_mask.png'), 1)
                                gmask = cv2.resize(gmask, (448, 576))
                                gmmask = cv2.applyColorMap(gmask, cv2.COLORMAP_RAINBOW) 
                                gmask = gmask[:, :, 0]
                                gmask[np.where(gmask!=0)] = 1
                                gmmask[np.where(gmask==0)] = 0
                                img = imgOriginal + gmmask*0.3 + hmmask*0.3
                                outname = os.path.join(OUTPUT_DIR, img_name+'_'+ ill_name_pre +'mask.png')       
                                cv2.imwrite(outname, img)
                                
                                hmask[np.where(hmask!=0)] = 1
                                if np.sum(hmask) == 0:
                                    continue
                                iobb = np.sum(hmask*gmask)/np.sum(hmask)   #ior
                                print(iobb)
        #                        iobb = np.sum(hmask*gmask)/(np.sum(hmask)+np.sum(gmask)-np.sum(hmask*gmask))  #iou
                                ior[ill_name]['ior'] += iobb
                                ior[ill_name]['count'] += 1
                                if iobb >= 0.1: acc[ill_name] += 1
                                elif iobb < 0.1: afp[ill_name] += 1                              
                                continue
                        
                            img = hmmask*0.3 + imgOriginal# + gmmask*0.3
                            hmask[np.where(hmask!=0)] = 1
                            hmmask[np.where(hmask==0)] = 0
                            outname = os.path.join(OUTPUT_DIR, img_name+'_'+ ill_name_pre +'.png')
                            cv2.imwrite(outname, img)       
                                
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
        print('The average of ACC is : {}'.format(ACC/N_CLASSES))
        print('The average of AFP is : {}'.format(AFP/N_CLASSES))
        print('The average of IOR is : {}'.format(IOR/N_CLASSES))