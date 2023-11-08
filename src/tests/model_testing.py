import torch
import cv2, os
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt


def annotated_predictions(image_dir, model, classes):
    # List all image files in the specified directory.
    image_files = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png')]
    
    # Iterate over all the images and do forward pass.
    for image_path in image_files:
        # Get the ground truth class name from the image path.
        gt_class_name = image_path.split(os.path.sep)[-1].split('.')[0]
        # Read the image and create a copy.
        image = cv2.imread(image_path)
        orig_image = image.copy()
        
        # Preprocess the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        image = transform(image)
        image = torch.unsqueeze(image, 0)
        image = image.cuda()
        
        # Forward pass through the image.
        with torch.no_grad():
            outputs = model(image.cuda())
            prob_scores = F.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted_class = torch.max(prob_scores, 1)
            pred_class_name = classes[predicted_class.item()]
        
        # Move the CUDA tensor to CPU before converting to NumPy.
        image = image.cpu()
        image = image.squeeze(0).permute(1, 2, 0).numpy()
        
        # Create a Matplotlib figure for image display.
        fig, ax = plt.subplots()
        ax.imshow(image)
        # Annotate the image with ground truth.
        ax.text(7, 15, f"GT: {gt_class_name}", color='purple', fontsize=12, weight='bold')
        # Annotate the image with prediction.
        ax.text(7, 32, f"Pred: {pred_class_name}", color='red', fontsize=12, weight='bold')
        plt.show()