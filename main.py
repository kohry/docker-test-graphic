from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import io  # io 모듈 추가

app = Flask(__name__)

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ResNet50 모델 로드
model = models.resnet50(pretrained=True)
model = model.to(device)
model.eval()  # 모델을 추론 모드로 설정

# 이미지 전처리 함수 정의
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # POST 요청에서 이미지를 받아옴
        image_data = request.files['image'].read()
        image = Image.open(io.BytesIO(image_data))
        image = preprocess_image(image)

        # 이미지를 모델에 대입하여 추론
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

        # 예측 결과 반환
        return jsonify({'class_id': predicted.item()})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
