import streamlit as st
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import requests

LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()


model = models.resnet18(pretrained=True)
model.eval()


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


st.title("Webcam Image Classification")
st.write("Real-time image classification using ResNet-18")


image_file = st.camera_input("Take a picture")

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Captured Image", use_column_width=True)

    
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)

    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    
    top5_prob, top5_idx = torch.topk(probabilities, 5)

    
    st.subheader("Top 5 Predictions")
    results = []
    for i in range(5):
        results.append({
            "Label": labels[top5_idx[i]],
            "Probability (%)": round(top5_prob[i].item() * 100, 2)
        })

    st.table(results)
