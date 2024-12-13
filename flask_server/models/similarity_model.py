def compute_similarity(file_path, genre):
    # 여기에 유사도 계산 로직 구현
    # 예시 데이터
    similarity_score = 0.85  # 임의의 유사도 점수
    similar_tracks = [
        {'id': '0eD9reMqWv79X3mAN41OhD', 'similarity': 0.9},
        {'id': '0eXf1Jmc2etXpzlt3dCh9d', 'similarity': 0.8},
        {'id': '1060gzllf4b0UETAXisR5l', 'similarity': 0.75},
        {'id': '0a4MMyCrzT0En247IhqZbD', 'similarity': 0.7},
        {'id': '0Fjf3IoSqPZMrk7zXl2oh4', 'similarity': 0.65},
        {'id': '0a4SMs889NwP8oWI7Vemle', 'similarity': 0.6},
    ]
    return similarity_score, similar_tracks
