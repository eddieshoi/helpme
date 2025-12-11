import streamlit as st
import tensorflow as tf
import keras
from keras import layers
from keras import ops
import numpy as np
from PIL import Image
import os

# ==========================================
# 0. 디자인 복구 (CSS 강제 주입)
# 원래 만드신 style.css 느낌을 내기 위해 스타일을 입힙니다.
# ==========================================
st.set_page_config(page_title="Shadow Play", page_icon="🌗", layout="wide")

st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&display=swap');

    /* 전체 폰트 및 배경 설정 */
    html, body, [class*="css"] {
        font-family: 'Pretendard', sans-serif;
        background-color: #ffffff;
        color: #1c1917;
    }
    
    /* 제목 스타일 (Playfair Display) */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        font-weight: 400;
    }
    
    /* 버튼 스타일 (검은색 모던한 버튼) */
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

    /* 파일 업로더 스타일 */
    .stFileUploader {
        border: 2px dashed #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }

    /* 상단 헤더 숨기기 (깔끔하게) */
    header {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 모델 부품 (Custom Layer)
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
# 2. 대용량 모델 다운로드 (구글 드라이브)
# ==========================================
import gdown

@st.cache_resource
def load_model_from_drive():
    # 🚨 여기에 아까 복사한 구글 드라이브 파일 ID를 넣으세요!
    file_id = '1QXUnKa3uCbK7kqgkXULYuEox0HGaE6hy' 
    
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'final_model.keras'
    
    if not os.path.exists(output):
        with st.spinner('모델 파일(248MB)을 다운로드 중입니다... 잠시만 기다려주세요.'):
            gdown.download(url, output, quiet=False)
    
    model = tf.keras.models.load_model(output, custom_objects={'Patches': Patches, 'PatchEncoder': PatchEncoder})
    return model

# ==========================================
# 3. 화면 구성 (원래 디자인 흉내)
# ==========================================

# 제목 섹션
st.markdown("<h1 style='font-size: 3rem; margin-bottom: 0;'>Light and Shadow,<br>Reading the Emotion Within.</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #4b5563; margin-bottom: 40px;'>AI-POWERED SHADOW ANALYSIS</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div style='border-top: 1px solid #e5e5e5; padding-top: 20px; margin-top: 20px;'>
        <p style='font-family: Playfair Display; font-style: italic; color: #9ca3af;'>Discover the unseen</p>
        <p style='line-height: 1.7; color: #4b5563;'>
            Every shadow tells a story. Shadow Play uses advanced AI to reveal the hidden emotional landscape within your images—transforming light and darkness into profound insight.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # 파일 업로더
    file = st.file_uploader("Upload Your Image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file).convert('RGB')
        st.image(image, use_container_width=True)
        
        # 모델 로드 시도
        try:
            # ID를 입력하지 않았으면 경고
            model = load_model_from_drive()
            
            if st.button("Analyze Emotion"):
                with st.spinner('Analyzing shadow contours...'):
                    img_array = image.resize((224, 224))
                    img_array = np.array(img_array).astype("float32") / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    logits = model(img_array, training=False)  # shape: (1, num_classes)
                    # 4. softmax로 확률 계산
                    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]  # (num_classes,)
                    class_names = ["calm", "cold", "lonely", "warm"]

                    # 5. 가장 확률 높은 클래스 + confidence
                    pred_class = int(np.argmax(probs))
                    confidence = float(probs[pred_class])
                    print(pred_class)
                    print(confidence)

                    print("predicted class index:", pred_class)
                    print("confidence:", confidence)

                    class_names = ["calm", "cold", "lonely", "warm"]
                    print("predicted label:", class_names[pred_class])
                    print("confidence:", float(confidence))

                    predictions = model.predict(img_array)
                    print(predictions)
                    probabilities = tf.nn.softmax(predictions).numpy()[0]
                    class_names = ["calm", "cold", "lonely", "warm"] # 순서 확인 필요
                    
                    idx = np.argmax(probabilities)
                    emotion = class_names[idx]
                    
                    # 결과 디자인 ###################### 이부분 class에 맞게 modify 필요합니다.
                    st.divider()
                    if emotion == 'calm':
                        st.markdown("<h2 style='color: #d97706;'>🍃 calm</h2>", unsafe_allow_html=True)
                        st.write("Radiant warmth and joy detected.")
                        st.audio("calm.m4a")
                    elif emotion == 'warm':
                        st.markdown("<h2 style='color: #dc2626;'>🔥 warm</h2>", unsafe_allow_html=True)
                        st.write("Strong energy and intensity detected.")
                        st.audio("calm.m4a")
                    elif emotion == 'cold':
                        st.markdown("<h2 style='color: #059669;'>🌞 cold</h2>", unsafe_allow_html=True)
                        st.write("Freezing cold.")
                        st.audio("calm.m4a")
                    elif emotion == 'lonely':
                        st.markdown("<h2 style='color: #059669;'>🌞 lonely</h2>", unsafe_allow_html=True)
                        st.write("Lonely.")
                        st.audio("calm.m4a")

        except Exception as e:
            st.error(f"오류가 발생했습니다. 구글 드라이브 ID를 코드에 넣었는지 확인해주세요: {e}")