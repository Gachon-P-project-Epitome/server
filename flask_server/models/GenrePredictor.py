import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.Processing import Preprocessing
from models.FeatureExtraction import FeatureExtracion
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.image import load_img, img_to_array



class GenrePredictor:
    def __init__(self, img, weights_file_path, vector_dir_path):
        self.img = img
        self.weights_file_path = weights_file_path
        self.vector_dir_path = vector_dir_path
        self.model = self.create_model()
        self.name_class = ["Electronic", "Experimental", "Folk", "Hip_Hop", "Instrumental", "International", "Pop", "Rock"]

        # GPU 사용 확인
        self.check_gpu_usage()
        
    def create_model(self):
        base_model_densenet = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        headModel = base_model_densenet.output
        headModel = Dropout(0.5)(headModel)
        headModel = GlobalAveragePooling2D()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Flatten()(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(1024, activation='relu')(headModel)
        headModel = Dense(8, activation='sigmoid')(headModel)  
        model_dense_121 = Model(inputs=base_model_densenet.input, outputs=headModel)
        model_dense_121.load_weights(self.weights_file_path)
        
        return model_dense_121
    
    def process_image(self):
        img = self.img.resize((224, 224), Image.LANCZOS).convert('RGB')  # RGB로 변환
        img_array = img_to_array(img)  # PIL 이미지를 NumPy 배열로 변환
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  
        return img_array
    
    def get_y_pred(self, avg_result):
        y_pred = []
        for sample in avg_result:
            toto = np.max(sample)
            y_pred.append([1.0 if (i == toto) else 0.0 for i in sample])
        y_pred = np.array(y_pred)
        return y_pred
    
    def extract_features(self, intermediate_layer_names):
        start_time3 = time.time()
        img_array = self.process_image()
        intermediate_models = [
            Model(inputs=self.model.input, outputs=self.model.get_layer(layer_name).output) 
            for layer_name in intermediate_layer_names
        ]
        
        intermediate_layer_names = [
            "conv2_block6_concat", 
            "conv3_block12_concat", 
            "conv4_block24_concat", 
            "conv5_block16_concat"
        ]
        
        feature_maps = {}
        for intermediate_model, layer_name in zip(intermediate_models, intermediate_layer_names):
            feature_map = intermediate_model.predict(img_array)
            feature_maps[layer_name] = feature_map

        print("Feature maps value: ")
        for layer_name, feature_map in feature_maps.items():
            print(f"{layer_name}: {feature_map.shape}") 

        all_features = np.concatenate([f.flatten() for f in feature_maps.values()], axis=0)
        all_features = all_features.reshape(1, -1)
        end_time3 = time.time()

        print(f"Time taken to feature extract time data: {end_time3 - start_time3:.4f} seconds")

        return all_features
    
    
    def predict_genre(self):
    
        img_array = self.process_image()
        start_time2 = time.time()
        result = self.model.predict(img_array)
        y_pred_res = self.get_y_pred(result)
        end_time2 = time.time()
        print(f"Time taken to model predict data: {end_time2 - start_time2:.4f} seconds")


        # 예측 확률에 따라 상위 2개의 장르 인덱스 선택
        top_indices = np.argsort(y_pred_res[0])[::-1][:2]  # 확률이 높은 두 개의 인덱스
        predicted_genres = [self.name_class[i] for i in top_indices]  # 인덱스를 장르로 변환

        print(f"Predicted genres: {predicted_genres}")

        
        features = []
        file_names = []

        for predicted_genre in predicted_genres:
            npz_file_path = os.path.join(self.vector_dir_path, f"{predicted_genre}.npz")
            
            if os.path.exists(npz_file_path):
                # 시간 측정 시작
                start_time = time.time()
                npz_data = np.load(npz_file_path, allow_pickle=True)
                # 시간 측정 종료
                print(f"Loaded {npz_file_path} successfully")
                start_time3 = time.time()
                features.append(npz_data['features'])
                end_time3 = time.time()
                start_time4 = time.time()
                file_names.append(npz_data['file_names'])
                end_time4 = time.time()
                end_time = time.time()
                print(f"Time taken to load {predicted_genre} data features: {end_time3 - start_time3:.4f} seconds")
                print(f"Time taken to load {predicted_genre} data file_names: {end_time4 - start_time4:.4f} seconds")
                print(f"Time taken to load {predicted_genre} data full: {end_time - start_time:.4f} seconds")
            else:
                print(f"NPZ file for predicted genre '{predicted_genre}' not found.")

        if features and file_names:
            return features, file_names  # 피처 + 이름 반환
        else:
            return None
    
    def check_gpu_usage(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            print(f"Available GPU(s): {physical_devices}")
        else:
            print("No GPU available.")