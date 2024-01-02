import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np

@st.cache_data
def load_image(uploaded_file, device):
    st.image(uploaded_file)
    image = np.array(Image.open(uploaded_file))
    # Preprocess the image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    image = torch.unsqueeze(transform(image), 0).to(device)
    return image


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def predict(image, model, classes):
    with torch.no_grad():
        outputs = model(image)
        prob_scores = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        _, predicted_class = torch.max(prob_scores, 1)
        pred_class_name = classes[predicted_class.item()]
    
    # Move the CUDA tensor to CPU before converting to NumPy.
    image = image.cpu()
    image = image.squeeze(0).permute(1, 2, 0).numpy()
    
    fig, ax = plt.subplots()
    ax.imshow(image)
    # Annotate the image with prediction.
    ax.text(7, 25, f"Pred: {pred_class_name}", color='red', fontsize=12, weight='bold')
    ax.axis(False)
    plt.grid(False)
    st.pyplot(fig)


@st.cache_data
def load_model(model_path, device):
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
    model = model.to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    return model
