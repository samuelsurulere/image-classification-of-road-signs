import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

plt.style.use('ggplot')


def train_imshow(train, classes, batch_size):
    dataiter = iter(train)
    images, labels = next(dataiter)
    print(images.shape)
    print(labels.shape)
    fig, axes = plt.subplots(figsize=(10, 4), ncols=batch_size)
    
    for i in range(batch_size):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0)) 
        ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
    plt.show()
    plt.savefig(f"/home/sammie/Capstone Project/image-classification-of-road-signs/output/figures/train_samples.png")


def test_imshow(test, classes, batch_size):
    dataiter = iter(test)
    images, labels = next(dataiter)
    print(images.shape)
    print(labels.shape)
    fig, axes = plt.subplots(figsize=(10, 4), ncols=batch_size)
    for i in range(batch_size):
        ax = axes[i]
        ax.imshow(images[i].permute(1, 2, 0)) 
        ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
    plt.show()
    plt.savefig(f"/home/sammie/Capstone Project/image-classification-of-road-signs/output/figures/test_samples.png")


def train_test_loss_function(n_epoch, epoch_loss_train, epoch_loss_test):
    fig, ax = plt.subplots()
    train_loss, = ax.plot(np.arange(0, n_epoch), epoch_loss_train, label='Training Loss')
    test_loss, = ax.plot(np.arange(0, n_epoch), epoch_loss_test, label='Testing Loss')
    ax.set_title('Training/testing loss')
    plt.legend(handles=[train_loss, test_loss])
    plt.show()
    plt.savefig(f"/home/sammie/Capstone Project/image-classification-of-road-signs/output/figures/loss.png")


def train_test_accuracy(n_epoch, epoch_acc_train, epoch_acc_test):
    fig, ax = plt.subplots()
    train_acc, = ax.plot(np.arange(0, n_epoch), epoch_acc_train, label=f"Training accuracy")
    test_acc, = ax.plot(np.arange(0, n_epoch), epoch_acc_test, label="Testing accuracy")
    ax.set_title('Training/testing accuracy')
    plt.legend(handles=[train_acc, test_acc])
    plt.show()
    plt.savefig(f"/home/sammie/Capstone Project/image-classification-of-road-signs/output/figures/accuracy.png")


def confusion_matrix_results(y, y_pred, classes):
    dataframe = pd.DataFrame(confusion_matrix(y, y_pred), index=classes, columns=classes)
    plt.figure(figsize=(10, 8))
    # Create heatmap
    sb.heatmap(dataframe,annot=True,cbar=None,cmap="YlGnBu",fmt="d")
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.show()
    plt.savefig(f"/home/sammie/Capstone Project/image-classification-of-road-signs/output/figures/confusion_matrix.png")


def model_classification_report(y, y_pred):
    return classification_report(y, y_pred, zero_division=0)