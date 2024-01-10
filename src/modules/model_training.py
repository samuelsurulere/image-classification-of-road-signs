# Importing necessary modules
import torchvision.models as models
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
# from tqdm import tqdm

# This function is necessary for efficientnet_b3 not to raise an exception
def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)
WeightsEnum.get_state_dict = get_state_dict


# Defining particular model to be used during training. 

model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
# model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freezing backbone and/or using maximum pooling during training
freeze_backbone = False
use_avg_pooling = False

if use_avg_pooling:
    maxpools = [k.split('.') for k, m in model.named_modules() if type(m).__name__ == 'AdaptiveAvgPool2d']
    for *parent, k in maxpools:
        setattr(model.get_submodule('.'.join(parent)),'avgpool', nn.AdaptiveMaxPool2d(output_size=1))

if freeze_backbone:
    for params in model.parameters():
        params.requires_grad = False

# Fine tuning final layer of the CNN model
model.classifier = nn.Sequential(
    nn.Linear(in_features=1536, out_features=512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=512, out_features=512),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(in_features=512, out_features=17)
)

# Learning rate of the model. Learning rate ratio and patience (used in the scheduler function).
lr = 0.0003
lr_ratio = 0.9
patience = 10

# loss function
criterion = nn.CrossEntropyLoss()

# optimizer function. Adam or SVD. L2 regularization can be introduced for Adam's optimizer

optimizer = optim.Adam(
                        params=model.parameters(), 
                        lr=lr, 
                        weight_decay=1e-2) # , weight_decay=1e-5
# optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=0.9)

# Scheduler function
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                            optimizer, 
                                            'max', 
                                            patience=patience, 
                                            factor=lr_ratio)


# Checking for availability of CUDA (GPU)
if torch.cuda.is_available():
    print("USING GPU")
    model = model.cuda()
    criterion = criterion.cuda()
else:
    print("USING CPU")
    model = model.cpu()
    criterion = criterion.cpu()


# Function for validation of the model
def test_epoch(test):
    
    correct = 0
    total = 0
    k = 0
    loss_total = 0
    
    with torch.no_grad():
        for data in test:
            images, labels = data
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            loss_k = criterion(outputs, labels.cuda())
            total_k = labels.size(0)
            total += total_k
            correct_k = np.sum(labels.numpy() == predicted.cpu().numpy())
            correct += correct_k
            k += 1
            loss_total += total_k*loss_k.item()
            
    test_accuracy = 100*correct/total
    test_loss = loss_total/total
    
    return test_loss, test_accuracy


# Function for training the model
def training_the_model(n_epoch, train, test):
    
    k = 0
    kstep = 1
    
    epoch_loss_train = np.zeros(n_epoch)
    epoch_acc_train = np.zeros(n_epoch)
    epoch_loss_test = np.zeros(n_epoch)
    epoch_acc_test = np.zeros(n_epoch)
    epoch_learning_rate = np.zeros(n_epoch)
    
    # loop over the dataset multiple times
    for epoch in range(n_epoch):  
        
        running_loss = 0.0
        running_n = 0
        running_acc = 0.0
        
        # model.train()
        
        # for i, data in tqdm(enumerate(train, 0), total=len(train)):
        for i, data in enumerate(train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            
            # zero the parameter gradients
            if k % kstep == 0:
                optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs.cuda())
            loss = criterion(outputs, labels.cuda())
            loss.backward()
            
            if k % kstep == 0:
                optimizer.step()
            
            running_n += len(outputs)
            
            labels_pred = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            
            acc = np.sum(labels_pred == labels.detach().numpy())
            
            running_loss += len(outputs)*loss.item()
            running_acc += acc
            
            k += 1
            
        loss_test, acc_test = test_epoch(test)
        
        scheduler.step(acc_test)
        
        sched_lr = optimizer.param_groups[0]['lr']
        
        print(f'epoch: {epoch:04d} | '
            f'training loss = {running_loss/running_n:0.4f} | '
            f'training accuracy = {100*running_acc/running_n:0.1f} | '
            f'test loss = {loss_test:0.4f} | '
            f'test accuracy = {acc_test:0.1f} | '
            f'learning rate = {sched_lr}')
        
        epoch_loss_train[epoch] = running_loss/running_n
        epoch_acc_train[epoch] = 100*running_acc/running_n
        epoch_loss_test[epoch] = loss_test
        epoch_acc_test[epoch] = acc_test
        epoch_learning_rate[epoch] = sched_lr
        
    torch.save(model.state_dict(), '../output/model_temp.pth')
    
    return epoch_loss_train, epoch_acc_train, epoch_loss_test, epoch_acc_test