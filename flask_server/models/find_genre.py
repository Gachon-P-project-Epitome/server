from io import BytesIO
from .GenrePredictor import GenrePredictor
import os
from models.Preprocessing import Preprocessing
from models.FeatureExtraction import FeatureExtracion

# base_dir = os.path.dirname(os.path.abspath(__file__))
# mp3_file_path1 = os.path.join(base_dir, 'models', 'music', 'upload.mp3')
# weights_file_path1 = os.path.join(base_dir, 'models', 'epoch_070_weights.h5')

def classify_genres2(file):
    mp3_data = file.read()  # BytesIO 객체에서 MP3 데이터 읽기
    #mp3_save_path = "/Users/habeomsu/epitome/flask_server/models/music/upload.mp3"  # 파일을 저장할 경로
    mp3_save_path = "/app/models/music/upload.mp3"
    with open(mp3_save_path, 'wb') as f:
        f.write(mp3_data)  # MP3 데이터를 파일에 저장
    
    preprocessing = Preprocessing(sr=16000)
    wav_path = mp3_save_path.replace(".mp3", ".wav")
    pcm_path = mp3_save_path.replace(".mp3", ".pcm")
    
    preprocessing.process_audio(mp3_save_path, wav_path, pcm_path)
    
    # Step 2: Feature Extraction (PCM to Mel Spectrogram Image)
    feature_extraction = FeatureExtracion()
    img = feature_extraction.mel_spectrogram(pcm_path)

    #weights_file_path ='/Users/habeomsu/epitome/flask_server/models/epoch_070_weights.h5' 
    weights_file_path ='/app/models/epoch_070_weights.h5'
    #vector_dir_path='/Users/habeomsu/epitome/flask_server/models/vector'
    vector_dir_path='/app/models/vector'
    img_path = mp3_save_path.replace(".mp3", ".png")  

    intermediate_layer_names = [
        "conv2_block6_concat",
        "conv3_block12_concat",
        "conv4_block24_concat",
        "conv5_block16_concat"
    ]
    
    genre_predictor = GenrePredictor(img_path=img_path,weights_file_path=weights_file_path,vector_dir_path=vector_dir_path)

    # Define the output image path based on the MP3 file name
    
    
    # Predict genre and get features
    predicted_genre = genre_predictor.predict_genre()


    # Clean up the uploaded file
    os.remove(mp3_save_path)
    os.remove(img_path)
    os.remove(wav_path)
    os.remove(pcm_path)

    # 결과 반환
    return predicted_genre

    

