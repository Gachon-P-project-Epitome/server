from io import BytesIO
from .test3 import MusicGenrePredictor
import os

# base_dir = os.path.dirname(os.path.abspath(__file__))
# mp3_file_path1 = os.path.join(base_dir, 'models', 'music', 'upload.mp3')
# weights_file_path1 = os.path.join(base_dir, 'models', 'epoch_070_weights.h5')

def classify_genres(file):
    mp3_data = file.read()  # BytesIO 객체에서 MP3 데이터 읽기
    #mp3_save_path = "/Users/habeomsu/epitome/flask_server/models/music/upload.mp3"  # 파일을 저장할 경로
    mp3_save_path = "/app/models/music/upload.mp3"
    with open(mp3_save_path, 'wb') as f:
        f.write(mp3_data)  # MP3 데이터를 파일에 저장

    #weights_file_path ='/Users/habeomsu/epitome/flask_server/models/epoch_070_weights.h5'  
    weights_file_path ='/app/models/epoch_070_weights.h5'
    genre_predictor = MusicGenrePredictor(model_weights_path=weights_file_path)

    # Define the output image path based on the MP3 file name
    img_path = mp3_save_path.replace(".mp3", ".png")  
    
    # Predict genre and get features
    predicted_genre, intermediate_features = genre_predictor.predict_genre_with_features(mp3_save_path, img_path)

    # Clean up the uploaded file
    os.remove(mp3_save_path)

    # 결과 반환
    return predicted_genre

    

