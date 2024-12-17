import numpy as np
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from .GenrePredictor import GenrePredictor


from io import BytesIO
import os
from models.Preprocessing import Preprocessing
from models.FeatureExtraction import FeatureExtracion

class CosineSimilaritys:
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
        predicted_genre_data,name = self.genre_predictor.predict_genre()

        if predicted_genre_data is not None:
            print("Shape of the extracted vector from the NPZ file:")
            print(predicted_genre_data.shape)

            cosine_similarities = cosine_similarity(all_features, predicted_genre_data)

            print("Cosine Similarities between the image vector and the predicted genre vectors:")
            print(cosine_similarities.shape)
            
            
            
            # max_similarity_idx = np.argmax(cosine_similarities)
            # max_similarity_value = cosine_similarities[0, max_similarity_idx]
            # print(f"Most similar vector index: {max_similarity_idx}")
            # print(f"Most similar vector id: {name[max_similarity_idx]}")
            # print(f"Highest cosine similarity value: {max_similarity_value}")

            # 상위 5개 유사도 인덱스와 값을 가져오기
            top_n = 5
            top_indices = np.argsort(cosine_similarities[0])[::-1][:top_n]  # 유사도 배열을 내림차순으로 정렬하고 상위 N개 선택

            # 인덱스와 유사도 값을 출력
            for i in range(top_n):
                index = top_indices[i]  # predicted_genre_data에서의 인덱스
                similarity = cosine_similarities[0][index]  # 해당 인덱스의 유사도 값
                print(f"Most similar vector index: {index}")  # predicted_genre_data의 인덱스
                print(f"Most similar vector name: {name[index]}")  # names에서 해당 인덱스의 이름
                print(f"Highest cosine similarity value: {similarity}")
            
            def remove_genre(track_name):
                name_class = ["Electronic", "Experimental", "Folk", "Hip_Hop", "Instrumental", "International", "Pop", "Rock"]
            # name_class에 있는 모든 장르를 제거
                for genre in name_class:
                    track_name = track_name.replace(genre, '')
                return track_name.replace('.png', '').strip()  # .png도 제거하고 공백도 제거
            
            def genre_output(track_name):
                name_class = ["Electronic", "Experimental", "Folk", "Hip_Hop", "Instrumental", "International", "Pop", "Rock"]
                for genre in name_class:
                    if genre in track_name:
                        return genre  # 장르가 존재할 경우 해당 장르를 반환
                return None



            
            # 유사한 트랙 구성
            genre=genre_output(name[0])
            print(genre)
           
            similar_tracks = [
                {
                    'id': remove_genre(name[index]),  # 실제 데이터 구조에 맞게 수정
                    'similarity': float(cosine_similarities[0][index])
                } for index in top_indices
            ]
            print(genre,similar_tracks)
            return genre,similar_tracks

        else:
            print("No NPZ data extracted.")
    
    


# def main():
#     #mp3_data = file.read()
#     #mp3_save_path = "/Users/habeomsu/epitome/flask_server/models/music/upload.mp3"  # 파일을 저장할 경로
#     mp3_save_path = "/Users/habeomsu/무제 폴더/0eD9reMqWv79X3mAN41OhD.mp3"

#     # with open(mp3_save_path, 'wb') as f:
#     #     f.write(mp3_data)  # MP3 데이터를 파일에 저장
    
#     preprocessing = Preprocessing(sr=16000)
#     wav_path = mp3_save_path.replace(".mp3", ".wav")
#     pcm_path = mp3_save_path.replace(".mp3", ".pcm")
    
#     preprocessing.process_audio(mp3_save_path, wav_path, pcm_path)
    
#     # Step 2: Feature Extraction (PCM to Mel Spectrogram Image)
#     feature_extraction = FeatureExtracion()
#     img = feature_extraction.mel_spectrogram(pcm_path)


#     img_path = mp3_save_path.replace(".mp3", ".png")
#     img.save(img_path) 

    
#     weights_file_path ='/Users/habeomsu/epitome/flask_server/models/epoch_070_weights.h5' 
#     #weights_file_path ='/app/models/epoch_070_weights.h5'
#     vector_dir_path='/Users/habeomsu/epitome/flask_server/models/vector'
#     #vector_dir_path='/app/models/vector'

#     intermediate_layer_names = [
#         "conv2_block6_concat",
#         "conv3_block12_concat",
#         "conv4_block24_concat",
#         "conv5_block16_concat"
#     ]

#     pipeline = CosineSimilaritys(img_path, weights_file_path, vector_dir_path)
#     all_features = pipeline.extract_features(intermediate_layer_names)
#     pipeline.predict_genre_and_calculate_similarity(all_features)

    
#     os.remove(img_path)
#     os.remove(wav_path)
#     os.remove(pcm_path)

    

# if __name__ == "__main__":
#     main()