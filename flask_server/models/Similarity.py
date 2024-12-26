import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .GenrePredictor import GenrePredictor


from io import BytesIO
from models.FeatureExtraction import FeatureExtracion

class CosineSimilaritys:
    def __init__(self, img, weights_file_path, vector_dir_path):
        self.img = img
        self.weights_file_path = weights_file_path
        self.vector_dir_path = vector_dir_path
        self.genre_predictor = self.initialize_genre_predictor()

    def initialize_genre_predictor(self):
        return GenrePredictor(self.img, self.weights_file_path, self.vector_dir_path)

    def extract_features(self, intermediate_layer_names):
        all_features = self.genre_predictor.extract_features(intermediate_layer_names)
        print("Extracted Features Shape:")
        print(all_features.shape)
        return all_features

    def predict_genre_and_calculate_similarity(self, all_features):
        predicted_genre_data, names = self.genre_predictor.predict_genre()

        if predicted_genre_data is not None:
            print("Shapes of the extracted vectors from the NPZ files:")
            for data in predicted_genre_data:
                print(data.shape)  # 각 NPZ 벡터의 shape 출력

            # 첫 번째 NPZ 벡터와 두 번째 NPZ 벡터에 대해 코사인 유사도 계산
            cosine_similarities_1 = cosine_similarity(all_features, predicted_genre_data[0])
            cosine_similarities_2 = cosine_similarity(all_features, predicted_genre_data[1])

                # 상위 5개 유사도 인덱스와 값을 가져오기
            top_n = 5
            top_results = []

            for i, cosine_similarities in enumerate([cosine_similarities_1, cosine_similarities_2]):
                top_indices = np.argsort(cosine_similarities[0])[::-1][:top_n]  # 유사도 배열을 내림차순으로 정렬하고 상위 N개 선택
                
                # 인덱스와 유사도 값을 출력
                for index in top_indices:
                    similarity = cosine_similarities[0][index]  # 해당 인덱스의 유사도 값
                    print(f"Most similar vector index for NPZ vector {i + 1}: {index}")  # predicted_genre_data의 인덱스
                    print(f"Most similar vector name: {names[i][index]}")  # names에서 해당 인덱스의 이름
                    print(f"Highest cosine similarity value: {similarity}")
                    
                    genre = self.genre_output(names[i][index])  # 장르 추출

                    # "HipPop" 변경
                    if genre == "HipPop":
                        genre = "Hip_Hop"

                    # 결과 저장
                    top_results.append({
                        'genre': genre,  # 장르 추가
                        'genre_index': i + 1,
                        'id': self.remove_genre(names[i][index]),  # 실제 데이터 구조에 맞게 수정
                        'similarity': float(similarity)
                    })

            # 가장 높은 유사도를 가진 항목 선택
            if top_results:
                

                # 상위 유사한 트랙 구성
                similar_tracks = [
                    {
                        'id': entry['id'],  # 실제 데이터 구조에 맞게 수정
                        'similarity': entry['similarity'],
                        'genre': entry['genre']
                    } for entry in top_results  # 상위 5개 유사한 트랙
                ]
                return similar_tracks  # 장르와 유사한 트랙 반환

        else:
            print("No NPZ data extracted.")
            return None, None  # 장르와 트랙 정보 모두 None 반환


    def remove_genre(self, track_name):
        name_class = ["Electronic", "Experimental", "Folk", "HipPop", "Instrumental", "International", "Pop", "Rock"]
        # name_class에 있는 모든 장르를 제거
        for genre in name_class:
            track_name = track_name.replace(genre, '')
        return track_name.replace('.png', '').strip()  # .png도 제거하고 공백도 제거

    def genre_output(self, track_name):
        name_class = ["Electronic", "Experimental", "Folk", "HipPop", "Instrumental", "International", "Pop", "Rock"]
        for genre in name_class:
            if genre in track_name:
                return genre  # 장르가 존재할 경우 해당 장르를 반환
        return None