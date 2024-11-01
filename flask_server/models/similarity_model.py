def compute_similarity(file_path, genre):
    # 여기에 유사도 계산 로직 구현
    # 예시 데이터
    similarity_score = 0.85  # 임의의 유사도 점수
    similar_tracks = [
        {'id': '37S86pw74OH8j96ZmMnrpR', 'similarity': 0.9},
        {'id': '6oNLSQX8bcAdbCElZYju3v', 'similarity': 0.8},
        {'id': '4fouWK6XVHhzl78KzQ1UjL', 'similarity': 0.75},
        {'id': '1lTBkwEm0wim9RsMXqtqWy', 'similarity': 0.7},
        {'id': '1SS0WlKhJewviwEDZ6dWj0', 'similarity': 0.65},
        {'id': '5vNRhkKd0yEAg8suGBpjeY', 'similarity': 0.6},
    ]
    return similarity_score, similar_tracks
