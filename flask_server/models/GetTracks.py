from io import BytesIO
import os
from models.Processing import Preprocessing
from models.FeatureExtraction import FeatureExtracion
from models.Similarity import CosineSimilaritys


def get_tracks(file):
    mp3_data = file.read()  # BytesIO 객체에서 MP3 데이터 읽기
   
    
    preprocessing = Preprocessing(sr=16000)
    
    pcm_data = preprocessing.process_audio(mp3_data)
    
    # Step 2: Feature Extraction (PCM to Mel Spectrogram Image)
    feature_extraction = FeatureExtracion()
    img = feature_extraction.mel_spectrogram(pcm_data)
    

    # weights_file_path ='/Users/habeomsu/epitome/flask_server/models/epoch_070_weights.h5' 
    weights_file_path ='/app/models/epoch_070_weights.h5'
    # vector_dir_path='/Users/habeomsu/epitome/flask_server/models/vector'
    vector_dir_path='/app/models/vector'
    

    intermediate_layer_names = [
        "conv2_block6_concat",
        "conv3_block12_concat",
        "conv4_block24_concat",
        "conv5_block16_concat"
    ]

    pipeline = CosineSimilaritys(img, weights_file_path, vector_dir_path)
    all_features = pipeline.extract_features(intermediate_layer_names)
    genre,similar_tracks=pipeline.predict_genre_and_calculate_similarity(all_features)

    
    # 결과 반환
    return genre,similar_tracks

    

