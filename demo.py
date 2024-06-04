from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from vit_pytorch import ViT
from torchvision.models import efficientnet_b1, efficientnet_b2
import numpy as np
import google.generativeai as genai
import threading


GOOGLE_API_KEY = "AIzaSyDVvRFIaAr9cCsOf6G0AQZHOqryuLswymU"
genai.configure(api_key=GOOGLE_API_KEY)
# model_api = genai.GenerativeModel('gemini-pro')
model_api = genai.GenerativeModel('gemini-1.5-pro-latest')
chat = model_api.start_chat(history=[])

# Flask 애플리케이션 및 기타 설정
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 4  # 감정 클래스 수

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 모델 정의: EfficientNet + ViT
class HybridModel(nn.Module):
    def __init__(self, num_classes, model_version='b1'):
        super(HybridModel, self).__init__()
        if model_version == 'b1':
            self.efficientnet = efficientnet_b1(pretrained=True)
        elif model_version == 'b2':
            self.efficientnet = efficientnet_b2(pretrained=True)
        else:
            raise ValueError("model_version should be 'b1' or 'b2'")
        
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Remove the fully connected layer
        
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=1024,  # Use 1024 as an intermediate dimension
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        self.fc = nn.Linear(num_ftrs + 1024, num_classes)  # Adjusted for combined features

    def forward(self, x):
        efficientnet_features = self.efficientnet(x)  # Shape: (batch_size, num_ftrs)
        vit_features = self.vit(x)  # Shape: (batch_size, 1024)
        combined_features = torch.cat((efficientnet_features, vit_features), dim=1)  # Shape: (batch_size, num_ftrs + 1024)
        out = self.fc(combined_features)
        return out

# 모델 초기화 및 가중치 로드
model_version = 'b1'  # 'b1' 또는 'b2'로 변경 가능
model = HybridModel(num_classes=num_classes, model_version=model_version).to(device)
model.load_state_dict(torch.load(f'best_hybrid_model_efficientnet_{model_version}.pth', map_location=device))
model.eval()

# OpenCV 웹캠 초기화
cap = cv2.VideoCapture(0)

# 감정 레이블
emotion_labels = ['anger', 'happy', 'panic', 'sadness']

# 전역 변수로 감정 상태 저장
global_emotions = []

# 감정 및 프레임 감지 함수
def detect_emotion(frame):
    global global_emotions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    emotions = []
    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = transform(face_img).unsqueeze(0).to(device)
        outputs = model(face_img)
        _, predicted = torch.max(outputs, 1)
        emotion = emotion_labels[predicted.item()]
        emotions.append(emotion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    global_emotions = emotions
    return emotions, frame

# 얼굴 감지기 로드
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 웹캠 프레임 생성기
def gen_frames():  
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        emotions, frame = detect_emotion(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 채팅 입력을 처리하는 함수
@app.route('/chat', methods=['POST'])
def chat_response():
    global global_emotions
    chat_input = request.form['message']
    if global_emotions:
        prompt = f"My emotion seems like {', '.join(global_emotions)}. {chat_input} \n\n 무조건 한국어로 대화하면 좋겠어요. 심리 상담가처럼 응답해줬으면 좋겠어요. 길게 정보를 나열하기 보다는 대화를 통해 자연스럽게 정보 전달이 되었으면 좋겠어요."
        response = chat.send_message(prompt)
        return jsonify({'response': response.text})
    else:
        return jsonify({'response': 'No emotion detected'})

@app.route('/emotion_video')
def emotion_video():
    global global_emotions
    if global_emotions:
        emotion = global_emotions[0]
        video_path = f"static/gifs/{emotion}_gif.gif"
        return jsonify({'video_path': video_path})
    else:
        return jsonify({'video_path': ''})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
