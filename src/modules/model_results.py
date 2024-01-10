import torch
import numpy as np


def model_predictions(test, model, criterion):
    
    correct = 0
    total = 0
    k = 0
    loss_total = 0
    
    y = []
    y_pred = []
    
    # model.eval()
    
    with torch.no_grad():
        for data in test:
            images, labels = data
            
            y += labels.numpy().tolist()
            
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs.data, 1)
            
            y_pred += predicted.cpu().numpy().tolist()
            
            loss_k = criterion(outputs, labels.cuda())
            total_k = labels.size(0)
            total += total_k
            correct_k = np.sum(labels.numpy() == predicted.cpu().numpy())
            correct += correct_k
            k += 1
            loss_total += total_k*loss_k.item()
            
    test_accuracy = 100*correct/total
    test_loss = loss_total/total
    
    return correct, total, test_accuracy, test_loss, y, y_pred