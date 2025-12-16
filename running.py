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
# 0. 디자인 설정
# ==========================================
st.set_page_config(page_title="Shadow Play", page_icon="🌗", layout="wide")

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
# 1. Custom Layers
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
# 2. 모델 로드
# ==========================================
@st.cache_resource
def load_model_from_drive():
    # 구글 드라이브 ID (유지)
    file_id = '1QXUnKa3uCbK7kqgkXULYuEox0HGaE6hy' 
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'final_model.keras'
    
    if not os.path.exists(output):
        with st.spinner('모델 파일(248MB)을 다운로드 중입니다... 잠시만 기다려주세요.'):
            gdown.download(url, output, quiet=False)
    
    model = tf.keras.models.load_model(output, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
    return model

# ==========================================
# 3. 메인 로직
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
                    # 1. 이미지 전처리 (핵심 수정!)
                    # / 255.0 을 제거했습니다. 0~255 범위의 값을 그대로 넣습니다.
                    img_array = image.resize((224, 224))
                    img_array = np.array(img_array).astype("float32") 
                    # img_array = img_array / 255.0  <-- 이 코드가 범인이었습니다! 삭제함.
                    
                    img_array = np.expand_dims(img_array, axis=0)

                    # 2. 예측
                    # Sigmoid 대신 다시 Softmax를 쓰거나, 모델의 마지막 층에 따라 다름
                    # 일단 logits 그대로 받아서 softmax로 확률화하는 것이 가장 일반적입니다.
                    predictions = model.predict(img_array)
                    probabilities = tf.nn.softmax(predictions).numpy()[0]
                    
                    # 🚨 클래스 이름 (알파벳 순서)
                    class_names = ["calm", "cold", "lonely", "warm"]
                    
                    # 3. 디버깅용 확률 출력 (결과가 이상하면 이 숫자를 보세요)
                    # st.write("각 감정별 확률:", {n: float(p) for n, p in zip(class_names, probabilities)})

                    idx = np.argmax(probabilities)
                    emotion = class_names[idx]
                    confidence = probabilities[idx]

                    # 4. 결과 출력
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
                        st.audio("happy.m4a") 
                    
                    st.caption(f"Confidence: {confidence*100:.2f}%")

        except Exception as e:
            st.error(f"오류가 발생했습니다.: {e}")
