import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from lib import  modules
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split #训练集，测试集划分函数
# Ignore warnings
import warnings
import copy
from scipy.io import savemat, loadmat
import torch.nn as nn
from collections import defaultdict
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary
import random
warnings.filterwarnings("ignore")

def find_files_with_suffix(directory, suffix):
    files_with_suffix = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                files_with_suffix.append(os.path.join(root, file))
    return files_with_suffix

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return [self.data[idx],self.label[idx]]
    
def calc_loss_dense(pred, target, metrics):
    criterion = nn.MSELoss()
    loss = criterion(pred, target)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    return loss

def calc_loss_test(pred1, pred2, target, metrics, error="MSE"):
    criterion = nn.MSELoss()
    if error=="MSE":
        loss1 = criterion(pred1, target)
        loss2 = criterion(pred2, target)
    else:
        loss1 = criterion(pred1, target)/criterion(target, 0*target)
        loss2 = criterion(pred2, target)/criterion(target, 0*target)
    metrics['loss first U'] += loss1.data.cpu().numpy() * target.size(0)
    metrics['loss second U'] += loss2.data.cpu().numpy() * target.size(0)
    return [loss1,loss2]

def print_metrics(metrics, epoch_samples, phase):
    outputs1 = []
    outputs2 = []
    for k in metrics.keys():
        outputs1.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs1)))
    
def train_model(model, optimizer, scheduler, num_epochs=50, WNetPhase="firstU", targetType="dense", num_samples=300):
    # WNetPhase: traine first U and freez second ("firstU"), or vice verse ("secondU").
    # targetType: train against dense images ("dense") or sparse measurements ("sparse")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("learning rate", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss_dense(outputs, targets, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), read_path+'model/Trained_Model_'+WNetPhase+'.pt')
    print('Best val loss: {:4f}'.format(best_loss))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
if __name__ == '__main__':
    
    mode = 'test' # 'train' or 'test'
    scene = 'test15_data'
    
    NAN_ratio_list = [0.8,0.85,0.9,0.93,0.95,0.97,0.99]
    SNR_list = [2,6,10,14,18,22]
    abs_path = 'E:\\zzh清华\\GDOP张量分析\\张量分解和稀疏重构\\0-电磁频谱重构\\论文\\code2\\'+scene+'\\tdoa1\\ViT\\'
    # abs_path = 'E:\\zzh清华\\GDOP张量分析\\张量分解和稀疏重构\\0-电磁频谱重构\\论文\\code2\\'+scene+'\\tdoa1\\ViT\\'
    
    # ------------------ratio分析模式，先将mode改为train进行训练，再改为test进行推理---------------------
    # for NAN_ratio in NAN_ratio_list: 
    #     read_path = abs_path+'fs_10_ratio_'+str(NAN_ratio)+'/'
    #     model_path = abs_path+'fs_10_ratio_'+str(NAN_ratio)+'/'
    # ------------------SNR分析模式，无需训练，只需mode改为test直接推理，推理模型来自于ratio=0.8---------------------
    for snr in SNR_list: # SNR分析模式
        read_path = abs_path+'fs_10_snr_'+str(snr)+'/'
        model_path = abs_path+'fs_10_ratio_0.8/'
    # -----------------------------------------------------------------------------------------------------
        
        if mode=='train':
            #1-matlab训练数据加载和预处理#################################
            path_data = read_path + 'data/'  # mat文件路径
            suffix = '.mat'
            files = find_files_with_suffix(path_data, suffix)
            data_all = []
            for i in range(len(files)):
                data_i = loadmat(files[i])  # 读取mat文件
                data_i = data_i['data']
                data_all.append(data_i)
                
            path_label = read_path + 'label/'  # mat文件路径
            suffix = '.mat'
            files = find_files_with_suffix(path_label, suffix)
            label_all = []
            for i in range(len(files)):
                label_i = loadmat(files[i])  # 读取mat文件
                label_i = label_i['label']
                label_all.append(label_i)
                
            # indices = random.sample(range(len(label_all)), int(len(label_all)/5))
            # new_a = [data_all[i] for i in indices]
            # new_b = [label_all[i] for i in indices] 
            # data_all = new_a
            # label_all = new_b

            x_train,x_test,y_train,y_test=train_test_split(data_all,label_all,test_size=0.2,random_state=22)
            x_train=torch.FloatTensor(x_train)
            x_train = x_train.unsqueeze(1) #单通道
            y_train=torch.FloatTensor(y_train)
            y_train = y_train.unsqueeze(1)
            x_test=torch.FloatTensor(x_test)
            x_test = x_test.unsqueeze(1) #单通道
            y_test=torch.FloatTensor(y_test)
            y_test = y_test.unsqueeze(1) #单通道
            # y_train = y_train.unsqueeze(-1)
            # y_test = y_test.unsqueeze(-1)
            train_data = MyDataset(x_train,y_train)
            test_data = MyDataset(x_test,y_test)
            # train_dataloader = DataLoader(train_data,batch_size=8)

            batch_size = 8
            dataloaders = {
                'train': DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda')),
                'val': DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0, generator=torch.Generator(device='cuda'))
            }

            #2-ViT训练#################################
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            torch.set_default_dtype(torch.float32)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            torch.backends.cudnn.enabled
            
            model = modules.MatrixReconstructionViT(
                img_size=x_train.shape[-1],
                patch_size=8,
                embed_dim=128,
                depth=4,
                num_heads=4
            ).to(device)
            # model =modules.RadioWNet(inputs=1,phase="firstU") # inputs通道数
            # model.cuda()
            # summary(model, input_size=(batch_size, 80,80))
            optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)
            model_save_path = read_path + 'model/'
            if not os.path.exists(model_save_path): #判断所在目录下是否有该文件名的文件夹
                os.mkdir(model_save_path)

            model = train_model(model, optimizer_ft, exp_lr_scheduler,num_epochs = 100)
            
            
        
        else:
            #1-matlab测试数据加载和预处理#################################
            path_data = read_path + 'testdata/'  # mat文件路径
            path_label = read_path + 'testlabel/'  # mat文件路径
            path_out = read_path + 'outdata/'
            if not os.path.exists(path_out): #判断所在目录下是否有该文件名的文件夹
                os.mkdir(path_out)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            model = modules.MatrixReconstructionViT(
                img_size=80,
                patch_size=8,
                embed_dim=128,
                depth=4,
                num_heads=4
            ).to(device)
            model.load_state_dict(torch.load(model_path+'model/Trained_Model_firstU.pt'))
            print("The model {} has been loaded!".format(model_path+'model/Trained_Model_firstU.pt'))
            model.to(device)
            model.eval()   # Set model to evaluate mode
            for dir_i in os.listdir(path_data):
                read_data_path_i = path_data + dir_i + '/'
                read_label_path_i = path_label + dir_i + '/'
                out_data_path_i = path_out + dir_i + '/'
                if not os.path.exists(out_data_path_i): #判断所在目录下是否有该文件名的文件夹
                    os.mkdir(out_data_path_i)
                suffix = '.mat'
                files_data = find_files_with_suffix(read_data_path_i, suffix)
                files_label = find_files_with_suffix(read_label_path_i, suffix)
                data_all,label_all = [],[]
                for i in range(len(files_data)):
                    data_i = loadmat(files_data[i])  # 读取mat文件
                    label_i = loadmat(files_label[i])  # 读取mat文件
                    data_i,label_i = data_i['data'],label_i['label']
                    data_all.append(data_i)
                    label_all.append(label_i)
                data_all,label_all=torch.FloatTensor(data_all),torch.FloatTensor(label_all)
                data_all,label_all = data_all.unsqueeze(1),label_all.unsqueeze(1) #单通道
                test_data = MyDataset(data_all,label_all)
                batch_size = 1
                test_dataloader = DataLoader(test_data,batch_size=batch_size)
                
                
                count = 1
                for inputs, targets in test_dataloader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        out_file = outputs.squeeze()
                        savemat(out_data_path_i+'res_matrix'+str(count)+'.mat',{'res_matrix':out_file.cpu().detach().numpy()})
                        count = count + 1
                    
            


            