import numpy as np
import os
import matplotlib.pyplot as plt
import sys 
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.test1 import Preprocessing
from models.test2 import FeatureExtracion
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Number of music genres for FMA
number_class = 8
batch_sz = 1  # 단일 이미지 예측을 위해 배치 크기 1로 설정

# 8 genres for FMA
name_class = ["Electronic", "Experimental", "Folk", "Hip_Hop", "Instrumental", "International", "Pop", "Rock"]
tags = np.array(name_class)

# ImageDataGenerator for rescaling
test_data_gen = ImageDataGenerator(rescale=1/255.)

# 현재 디렉토리 기준으로 MP3 파일 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))
mp3_file_path = os.path.join(base_dir, 'models', 'music', 'upload.mp3')

# # 단일 MP3 파일 경로 
# mp3_file_path = "/Users/habeomsu/epitome/flask_server/models/music/upload.mp3"

class MusicGenrePredictor:
    def __init__(self, model_weights_path, input_size=(224, 224, 3), num_classes=8):
        self.model_weights_path = model_weights_path
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build the model
        self.model = self.model()
        
        # Load the trained weights
        self.model.load_weights(self.model_weights_path)

        # Intermediate models for feature extraction
        self.intermediate_layer_names = [
            "conv2_block6_concat",
            "conv3_block12_concat",
            "conv4_block24_concat",
            "conv5_block16_concat"
        ]
        self.intermediate_models = {
            layer_name: Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output)
            for layer_name in self.intermediate_layer_names
        }

    def model(self):
        """Builds the DenseNet121 model."""
        base_model_densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=self.input_size)     
        
        headModel = base_model_densenet.output
        headModel = Dropout(0.5)(headModel)
        headModel = GlobalAveragePooling2D()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Dense(self.num_classes, activation='sigmoid')(headModel)  # Assuming 8 classes for music genres
        model = Model(inputs=base_model_densenet.input, outputs=headModel)
        return model

    def predict_genre_with_features(self, mp3_path, output_img_path):
        """Predicts the genre and extracts intermediate features for a given MP3 file."""
        # Step 1: Preprocessing (MP3 to PCM to Mel Spectrogram Image)
        preprocessing = Preprocessing(sr=16000)
        wav_path = mp3_path.replace(".mp3", ".wav")
        pcm_path = mp3_path.replace(".mp3", ".pcm")
        
        preprocessing.process_audio(mp3_path, wav_path, pcm_path)
        
        # Step 2: Feature Extraction (PCM to Mel Spectrogram Image)
        feature_extraction = FeatureExtracion()
        img = feature_extraction.mel_spectrogram(pcm_path)

        os.remove(wav_path)
        os.remove(pcm_path)
        
        
        if img is None:
            print(f"Skipping {mp3_path} because the duration is too short.")
            return None
        
        
        # Step 3: Model prediction and feature extraction
        img_resized = np.resize(np.array(img), (224, 224, 3))  # 모델의 입력 크기에 맞게 이미지 리사이즈
        img_input = np.expand_dims(img_resized, axis=0)  # 배치 차원 추가
        
        # Predict the genre
        result = self.model.predict(img_input)
        y_pred_res = self._get_y_pred(result)
        y_pred = np.where(y_pred_res == 1)[1]
        predicted_genre = name_class[y_pred[0]]
        
        # Extract intermediate features
        intermediate_features = {
            layer_name: model.predict(img_input)
            for layer_name, model in self.intermediate_models.items()
        }
        
        return predicted_genre, intermediate_features

    def _get_y_pred(self, result):
        """Converts the model output to a binary vector."""
        return (result > 0.5).astype(int)

    def predict_for_single_song(self):
        """Predicts the genre for the single MP3 file."""
        # Step 1: Define the output image path based on the MP3 file name
        img_path = mp3_file_path.replace(".mp3", ".png")  
        
        # Step 2: Predict the genre and extract features for the single MP3 file
        predicted_genre, intermediate_features = self.predict_genre_with_features(mp3_file_path, img_path)  
        
        if predicted_genre:
            print(f"Predicted Genre for {mp3_file_path}: {predicted_genre}")
            for layer_name, features in intermediate_features.items():
                print(f"Intermediate Features from {layer_name}: Shape {features.shape}")
                print(features)

# # Instantiate and run the predictor
# weights_file_path = '/Users/habeomsu/epitome/flask_server/models/epoch_070_weights.h5'  

weights_file_path = os.path.join(base_dir, 'models', 'epoch_070_weights.h5')
