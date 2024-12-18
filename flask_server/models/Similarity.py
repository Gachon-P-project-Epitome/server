import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from GenrePredictor import GenrePredictor


from io import BytesIO
import os
from models.Preprocessing import Preprocessing
from models.FeatureExtraction import FeatureExtracion

class CosineSimilarity:
    def __init__(self, img_path, weights_file_path, vector_dir_path):
        self.img_path = img_path
        self.weights_file_path = weights_file_path
        self.vector_dir_path = vector_dir_path
        self.genre_predictor = self.initialize_genre_predictor()

    def initialize_genre_predictor(self):
        return GenrePredictor(self.img_path, self.weights_file_path, self.vector_dir_path)

    def extract_features(self, intermediate_layer_names):
        all_features = self.genre_predictor.extract_features(intermediate_layer_names)
        print("Extracted Features Shape:")
        print(all_features.shape)
        return all_features

    def predict_genre_and_calculate_similarity(self, all_features):
        predicted_genre_data = self.genre_predictor.predict_genre()

        if predicted_genre_data is not None:
            print("Shape of the extracted vector from the NPZ file:")
            print(predicted_genre_data.shape)

            cosine_similarities = cosine_similarity(all_features, predicted_genre_data)

            print("Cosine Similarities between the image vector and the predicted genre vectors:")
            print(cosine_similarities.shape)
            
            
            
            max_similarity_idx = np.argmax(cosine_similarities)
            max_similarity_value = cosine_similarities[0, max_similarity_idx]
            print(f"Most similar vector index: {max_similarity_idx}")
            print(predicted_genre_data[687])
            print(f"Highest cosine similarity value: {max_similarity_value}")
        else:
            print("No NPZ data extracted.")
