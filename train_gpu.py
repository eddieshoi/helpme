import os

# 1. GPU Setup
os.environ["KERAS_BACKEND"] = "tensorflow"
# XLA 관련 오류 방지를 위한 설정
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 2. Check GPU Availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ SUCCESS GPU Found: {gpus}")
else:
    print("⚠️ WARNING No GPU found. Training will be slow on CPU.")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DATA_PATH = "./dataset" 

TRAIN_DIR = os.path.join(BASE_DATA_PATH, "train")
TEST_DIR = os.path.join(BASE_DATA_PATH, "test")

BATCH_SIZE = 16 
LEARNING_RATE = 0.001
# [수정됨] 에포크를 50으로 증가 (EarlyStopping이 있으므로 충분히 크게 설정)
NUM_EPOCHS = 50
IMAGE_SIZE = 72
PATCH_SIZE = 6
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 8

# ==========================================
# DATA LOADING
# ==========================================
def load_dataset():
    if not os.path.exists(TRAIN_DIR):
        print(f"❌ Error Data path not found: {TRAIN_DIR}")
        return None, None, None

    print("Loading datasets...")
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=(224, 224),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=True
    )
    
    class_names = train_ds.class_names
    
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        image_size=(224, 224),
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds, val_ds, class_names

# ==========================================
# MODEL DEFINITION
# ==========================================
def get_augmenter():
    return keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(
            patches,
            (batch_size, num_patches_h * num_patches_w, self.patch_size * self.patch_size * channels),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        positions = tf.expand_dims(positions, axis=0)
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return config

def create_vit_classifier(num_classes):
    inputs = keras.Input(shape=(224, 224, 3))
    
    augmented = get_augmenter()(inputs)
    patches = Patches(PATCH_SIZE)(augmented)
    encoded_patches = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)

    for _ in range(TRANSFORMER_LAYERS):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        x3 = layers.Dense(PROJECTION_DIM * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(PROJECTION_DIM, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    
    features = layers.Dense(2048, activation=tf.nn.gelu)(representation)
    features = layers.Dropout(0.5)(features)
    features = layers.Dense(1024, activation=tf.nn.gelu)(features)
    features = layers.Dropout(0.5)(features)
    
    logits = layers.Dense(num_classes)(features)
    
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    train_ds, val_ds, class_names = load_dataset()
    
    if train_ds:
        num_classes = len(class_names)
        print(f"Detected {num_classes} classes: {class_names}")
        
        print("Adapting normalization layer...")
        temp_augmenter = get_augmenter()
        for images, _ in train_ds.take(1):
            temp_augmenter.layers[0].adapt(images)
            
        model = create_vit_classifier(num_classes)
        model.get_layer("data_augmentation").layers[0].set_weights(
            temp_augmenter.layers[0].get_weights()
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        
        model.summary()

        # [수정됨] EarlyStopping 콜백 정의
        # val_loss가 5번의 에포크 동안 개선되지 않으면 학습을 멈추고 가장 좋았던 모델 상태로 복원합니다.
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        print("\n🚀 Starting Training on RTX 4060 Ti...")
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=NUM_EPOCHS,
            callbacks=[early_stopping] # [수정됨] 콜백 추가
        )
        
        if not os.path.exists("models"):
            os.makedirs("models")
        
        # Save model with new name
        model.save("models/final_model.keras")
        print("\n💾 Model saved to models/final_model.keras")