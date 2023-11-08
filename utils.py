import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import cv2
from PIL import Image
import numpy as np

@st.cache_data
def load_image(uploaded_file):
    st.image(uploaded_file)
    image = np.array(Image.open(uploaded_file))
    # Preprocess the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = torch.unsqueeze(transform(image), 0)
    return image


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def predict(image, model, classes):
    with torch.no_grad():
        outputs = model(image.cuda())
        prob_scores = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        _, predicted_class = torch.max(prob_scores, 1)
        pred_class_name = classes[predicted_class.item()]
    
    # Move the CUDA tensor to CPU before converting to NumPy.
    image = image.cpu()
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    # Annotate the image with prediction.
    ax.text(7, 15, f"Pred: {pred_class_name}", color='purple', fontsize=12, weight='bold')
    st.pyplot(fig)


@st.cache_data
def load_model(model_path):
    model = models.efficientnet_b3()
    model.classifier = nn.Sequential(
        # nn.BatchNorm1d(num_features=1536, momentum=0.95),
        nn.Linear(in_features=1536, out_features=512),
        nn.ReLU(),
        # nn.Dropout(0.3),
        # nn.BatchNorm1d(num_features=512, momentum=0.95),
        nn.Linear(in_features=512, out_features=512),
        nn.ReLU(),
        # nn.Dropout(0.3),
        nn.Linear(in_features=512, out_features=13),
        nn.Softmax(dim=-1)
    )    
    if torch.cuda.is_available():
        model = model.cuda()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict, strict=False)
    return model
