
from io import BytesIO
import os
from models.Processing import Preprocessing
from models.FeatureExtraction import FeatureExtracion
from models.Similarity import CosineSimilaritys
from models.Similarity_GPU import CosineSimilaritys_GPU
import tensorflow as tf  # TensorFlow 추가


# 세션 초기화
tf.keras.backend.clear_session()

# GPU 메모리 성장 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    print("GPU 메모리 성장 설정 완료.")
else:
    print("GPU가 사용 가능하지 않습니다.")

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

   # GPU 사용 여부 확인 및 파이프라인 설정
    is_gpu_available = len(physical_devices) > 0

    

    
    # if is_gpu_available:
    #     print("Using GPU for similarity calculation.")
    #     pipeline2 = CosineSimilaritys_GPU(img, weights_file_path, vector_dir_path)
    #     all_features2 = pipeline2.extract_features(intermediate_layer_names)
    #     similar_tracks2=pipeline2.predict_genre_and_calculate_similarity(all_features2)
    #     print(similar_tracks2)
    #     # 결과 반환
    #     return  similar_tracks2
    
   
    pipeline = CosineSimilaritys(img, weights_file_path, vector_dir_path)
    all_features = pipeline.extract_features(intermediate_layer_names)
    similar_tracks=pipeline.predict_genre_and_calculate_similarity(all_features)
    print(similar_tracks)
    return  similar_tracks
        
    
    

    

