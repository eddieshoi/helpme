import streamlit as st
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np
from PIL import Image
import os
import gdown

# ==========================================
# 0. 페이지 설정 & 고대비 모드 스위치
# ==========================================
st.set_page_config(page_title="Shadow Play", page_icon="🌗", layout="wide")

# 상단에 스위치 배치를 위한 컬럼 분할 (오른쪽 구석에 배치)
top_col1, top_col2 = st.columns([10, 2])
with top_col2:
    high_contrast_on = st.toggle("High Contrast Mode")

# ==========================================
# 0-1. CSS 디자인 (스위치 상태에 따라 변경)
# ==========================================
if high_contrast_on:
    # [고대비 모드] 검은 배경 + 형광 노랑 글씨
    st.markdown("""
    <style>
        /* 전체 배경 및 폰트 강제 적용 */
        .stApp {
            background-color: #000000 !important;
            color: #FFFF00 !important;
        }
        
        /* 모든 텍스트 요소를 형광 노랑으로 강제 변환 */
        h1, h2, h3, p, div, span, label, .stMarkdown {
            color: #FFFF00 !important;
            font-family: sans-serif !important;
        }
        
        /* 버튼 스타일 (검정 배경/노랑 테두리) */
        .stButton > button {
            background-color: #000000 !important;
            color: #FFFF00 !important;
            border: 2px solid #FFFF00 !important;
            border-radius: 10px !important;
            font-weight: bold !important;
        }
        .stButton > button:hover {
            background-color: #FFFF00 !important;
            color: #000000 !important;
        }
        
        /* 파일 업로더 테두리 */
        .stFileUploader {
            border: 2px dashed #FFFF00 !important;
        }
        
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

else:
    # [기존 디자인] 원래 쓰시던 하얀색 깔끔한 스타일 (그대로 유지)
    st.markdown("""
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');

        html, body, [class*="css"] {
            font-family: 'Pretendard', sans-serif;
            background-color: #ffffff;
            color: #1c1917;
        }
        h1, h2, h3 {
            font-family: 'Playfair Display', serif !important;
            font-weight: 400;
        }
        .stButton > button {
            background-color: #111111 !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 10px 30px !important;
            border: none !important;
            transition: transform 0.2s;
        }
        .stButton > button:hover {
            transform: scale(1.02);
            background-color: #333 !important;
        }
        .stFileUploader {
            border: 2px dashed #e5e7eb;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
        }
        header {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. Custom Layers (수정 없음)
# ==========================================
@keras.saving.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size=6, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(patches, (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels))
        return patches
    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

@keras.saving.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=144, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)
    def call(self, patch):
        positions = ops.expand_dims(ops.arange(start=0, stop=self.num_patches, step=1), axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded
    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

# ==========================================
# 2. 모델 로드 (수정 없음)
# ==========================================
@st.cache_resource
def load_model_from_drive():
    # 구글 드라이브 ID
    file_id = '1QXUnKa3uCbK7kqgkXULYuEox0HGaE6hy' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'final_model.keras'
    
    if not os.path.exists(output):
        with st.spinner('모델 파일(248MB)을 다운로드 중입니다... 잠시만 기다려주세요.'):
            gdown.download(url, output, quiet=False)
    
    model = tf.keras.models.load_model(output, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
    return model

# ==========================================
# 3. 메인 로직 (수정 없음)
# ==========================================
st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>For Visually Impaired,<br>Reading the Emotion Within.</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #4b5563; margin-bottom: 40px;'>AI-POWERED SHADOW ANALYSIS</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div style='border-top: 1px solid #e5e5e5; padding-top: 20px; margin-top: 20px;'>
        <p style='font-family: Playfair Display; font-style: italic; color: #9ca3af;'>Discover the unseen</p>
        <p style='line-height: 1.7; color: #4b5563;'>
            Every shadow tells a story. Shadow Play uses advanced AI to reveal the hidden emotional landscape within your images.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    file = st.file_uploader("Upload Your Image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)
        
        try:
            model = load_model_from_drive()
            
            if st.button("Analyze Emotion"):
                with st.spinner('Analyzing shadow contours...'):
                    # 1. 이미지 전처리 (수정 안 함: 0~255 유지)
                    img_array = image.resize((224, 224))
                    img_array = np.array(img_array).astype("float32") 
                    
                    img_array = np.expand_dims(img_array, axis=0)

                    # 2. 예측
                    predictions = model.predict(img_array)
                    probabilities = tf.nn.softmax(predictions).numpy()[0]
                    
                    # 🚨 클래스 이름 (알파벳 순서)
                    class_names = ["calm", "cold", "lonely", "warm"]
                    
                    idx = np.argmax(probabilities)
                    emotion = class_names[idx]
                    confidence = probabilities[idx]

                    # 3. 결과 출력
                    st.divider()
                    if emotion == 'calm':
                        st.markdown("<h2 style='color: #d97706;'>🍃 calm</h2>", unsafe_allow_html=True)
                        st.write("Radiant warmth and joy detected.")
                        st.audio("calm.m4a")
                    elif emotion == 'cold':
                        st.markdown("<h2 style='color: #dc2626;'>🔥 cold</h2>", unsafe_allow_html=True)
                        st.write("Freezing cold.")
                        st.audio("sad.m4a") 
                    elif emotion == 'lonely':
                        st.markdown("<h2 style='color: #059669;'>🌑 lonely</h2>", unsafe_allow_html=True)
                        st.write("Lonely.")
                        st.audio("sad.m4a") 
                    elif emotion == 'warm':
                        st.markdown("<h2 style='color: #ea580c;'>🌞 warm</h2>", unsafe_allow_html=True)
                        st.write("Strong energy and intensity detected.")
                        st.audio("warm.m4a") 
                    
                    st.caption(f"Confidence: {confidence*100:.2f}%")

        except Exception as e:
            st.error(f"오류가 발생했습니다.: {e}")
