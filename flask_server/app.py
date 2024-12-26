from flask import Flask, request, jsonify
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# sys.path.append("/Users/habeomsu/epitome/flask_server")
from models.GetTracks import get_tracks

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
    

    similar_tracks = get_tracks(file)

    # 결과 구성
    results = {
        'original_file_id': file.filename,  # 실제 파일 ID를 사용할 수 있음
        'similar_tracks': [
            {
                'track_id': track['id'],
                'similarity': track['similarity'],
                'genre': track['genre'],
            } for track in similar_tracks
        ]
    }
    # Spring으로 결과 반환
    return jsonify(results), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
