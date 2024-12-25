import numpy as np
import pandas as pd
from PIL import Image
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from models.Processing import Preprocessing
from models.FeatureExtraction import FeatureExtracion
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
        
        return all_features
    
    
    def predict_genre(self):
    
        img_array = self.process_image()
        result = self.model.predict(img_array)
        y_pred_res = self.get_y_pred(result)

        # 예측 확률에 따라 상위 2개의 장르 인덱스 선택
        top_indices = np.argsort(y_pred_res[0])[::-1][:2]  # 확률이 높은 두 개의 인덱스
        predicted_genres = [self.name_class[i] for i in top_indices]  # 인덱스를 장르로 변환

        print(f"Predicted genres: {predicted_genres}")

        
        features = []
        file_names = []

        for predicted_genre in predicted_genres:
            npz_file_path = os.path.join(self.vector_dir_path, f"{predicted_genre}.npz")
            
            if os.path.exists(npz_file_path):
                npz_data = np.load(npz_file_path, allow_pickle=True)
                print(f"Loaded {npz_file_path} successfully")
                features.append(npz_data['features'])
                file_names.append(npz_data['file_names'])
            else:
                print(f"NPZ file for predicted genre '{predicted_genre}' not found.")

        if features and file_names:
            return features, file_names  # 피처 + 이름 반환
        else:
            return None
        