from flask import Flask, request, jsonify
from models.similarity_model import compute_similarity
from models.genre_classification_model import classify_genre
from utils.audio_processing import process_audio
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/process_music', methods=['POST'])
def process_music():
    # Spring으로부터 음악 파일 받기
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # 오디오 파일 처리 및 장르 분류
    genre = classify_genre(file)  # 파일 경로가 아닌 파일 객체를 직접 전달

    # 유사도 계산
    similarity_score, similar_tracks = compute_similarity(file, genre)  # 파일 객체를 전달

    # 결과 구성
    results = {
        'original_file_id': file.filename,  # 실제 파일 ID를 사용할 수 있음
        'similarity_score': similarity_score,
        'similar_tracks': [
            {
                'track_id': track['id'],
                'similarity': track['similarity'],
            } for track in similar_tracks
        ]
    }
    # Spring으로 결과 반환
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
