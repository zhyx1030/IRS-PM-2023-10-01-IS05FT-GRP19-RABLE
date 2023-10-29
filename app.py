import os
import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from test import *


app = Flask(__name__)
CORS(app)
# 初始化预训练的ResNet模型
model = models.resnet50(pretrained=True)
model.eval()

cifar10_classes = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]


# 路由用于接受图像上传和进行分类
def generate_random_image():
    random_image = np.random.randint(0, 256, size=(224, 224, 3), dtype=np.uint8)
    return Image.fromarray(random_image)

@app.route('/')
def index():
    return "Welcome to the website"

@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'})

    image_path = os.path.join('uploads', image.filename)
    image.save(image_path)

    # 加载并处理图像

    #img = Image.open(image_path)
    #img = generate_random_image()

    '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = preprocess(img)
    img = img.unsqueeze(0)  # 增加批次维度

    # 使用ResNet模型进行分类
    with torch.no_grad():
        output = model(img)

    # 解码分类结果
    _, indices = torch.topk(output, 5)
    categories = [cifar10_classes[boom(i)] for i in indices[0]]
    confidences = [output[0][i].item() for i in indices[0]]

    # 将分类结果返回给前端
    classification_result = {}
    for category, confidence in zip(categories, confidences):
        classification_result[category] = confidence

    
    '''
    classification_result = main(image_path)
    return jsonify(classification_result)

if __name__ == '__main__':
    app.run()
