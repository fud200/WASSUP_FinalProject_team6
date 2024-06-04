import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from vit_pytorch import ViT
from torchvision.models import efficientnet_b1, efficientnet_b2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from facenet_pytorch import MTCNN
from flask import Flask, render_template, Response, request, jsonify
import google.generativeai as genai

# 감정 레이블
emotion_labels = ['anger', 'happy', 'panic', 'sadness']

# Flask 애플리케이션 초기화
app = Flask(__name__)

# 데이터 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hybrid Model 정의: EfficientNet + ViT
class HybridModel(nn.Module):
    def __init__(self, num_classes, model_version='b1'):
        super(HybridModel, self).__init__()
        if model_version == 'b1':
            self.efficientnet = efficientnet_b1(pretrained=False)
        elif model_version == 'b2':
            self.efficientnet = efficientnet_b2(pretrained=False)
        else:
            raise ValueError("model_version should be 'b1' or 'b2'")
        
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Fully connected layer 제거
        
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=1024,  # 중간 차원으로 1024 사용
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )
        
        self.fc = nn.Linear(num_ftrs + 1024, num_classes)  # 결합된 특성에 맞게 조정

    def forward(self, x):
        efficientnet_features = self.efficientnet(x)  # Shape: (batch_size, num_ftrs)
        vit_features = self.vit(x)  # Shape: (batch_size, 1024)
        combined_features = torch.cat((efficientnet_features, vit_features), dim=1)  # Shape: (batch_size, num_ftrs + 1024)
        out = self.fc(combined_features)
        return out

# 모델 초기화 및 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_version = 'b1'  # 'b1' 또는 'b2'로 변경 가능
model = HybridModel(num_classes=len(emotion_labels), model_version=model_version).to(device)
model.load_state_dict(torch.load(f'best_hybrid_model_efficientnet_{model_version}.pth', map_location=device))
model.eval()

# MTCNN 초기화
mtcnn = MTCNN(keep_all=True, device=device)

# Google AI API 키
GOOGLE_API_KEY = "AIzaSyDVvRFIaAr9cCsOf6G0AQZHOqryuLswymU"

# GenAI 모델 초기화
genai.configure(api_key=GOOGLE_API_KEY)
model_api = genai.GenerativeModel('gemini-1.5-pro-latest')
chat = model_api.start_chat(history=[])

# 전역 변수로 감정 상태 저장
global_emotions = []

# 웹캠에서 프레임 생성하는 함수
def gen_frames():  
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # OpenCV에서 사용하는 BGR 색상을 RGB로 변환하여 PIL 이미지로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # 얼굴 감지
        boxes, _ = mtcnn.detect(pil_image)
        
        # 얼굴 주위에 사각형 그리기 및 감정 분석
        emotions = []  # 감정을 저장할 리스트
        if boxes is not None:
            draw = ImageDraw.Draw(pil_image)
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                
                # 얼굴 이미지를 추출하고 감정 분석 모델에 입력하기 위해 변환
                x, y, w, h = [int(coord) for coord in box]
                face_img = frame_rgb[y:y+h, x:x+w]
                face_img = Image.fromarray(face_img)  # NumPy 배열을 PIL 이미지로 변환
                face_img = transform(face_img).unsqueeze(0).to(device)
                
                # 감정 분석 모델로 예측
                outputs = model(face_img)
                _, predicted = torch.max(outputs, 1)
                emotion = emotion_labels[predicted.item()]
                
                # 얼굴 주위에 감정 표시
                font = ImageFont.truetype("arial.ttf", 30)  # 글꼴과 크기 설정
                draw.text((x, y), emotion, fill=(0, 255, 0, 255), font=font)
                
                # 감정을 리스트에 추가
                emotions.append(emotion)
        
        # 전역 변수에 감정을 저장
        global global_emotions
        global_emotions = emotions
        
        # OpenCV용 이미지로 변환하여 화면에 표시
        frame_drawn = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', frame_drawn)
        frame_drawn = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_drawn + b'\r\n')

# 채팅 입력을 처리하는 함수
@app.route('/chat', methods=['POST'])
def chat_response():
    global global_emotions
    chat_input = request.form['message']
    if global_emotions:
        emotions_str = ', '.join(global_emotions)
        prompt = f"My emotion seems like {emotions_str}. {chat_input} \n\n 무조건 한국어로 대화하면 좋겠어요. 심리 상담가처럼 응답해줬으면 좋겠어요. 길게 정보를 나열하기 보다는 대화를 통해 자연스럽게 정보 전달이 되었으면 좋겠어요."
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
