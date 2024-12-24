import subprocess
import librosa
import soundfile as sf
import numpy as np
import os
import io

class Preprocessing:
    def __init__(self, sr=16000):
        self.sr = sr

    def process_audio(self, mp3_file):
        """Converts MP3 to WAV and PCM in memory."""
        # Step 1: Convert MP3 to WAV in memory
        wav_data = self.wav_convert(mp3_file)

        # Step 2: Convert WAV to PCM in memory
        pcm_data = self.pcm_convert(wav_data)

        return pcm_data

    def wav_convert(self, mp3_file):
        """Converts MP3 file (BytesIO or bytes) to WAV (BytesIO)."""
        # mp3_file이 바이트 객체인 경우 BytesIO로 변환
        if isinstance(mp3_file, bytes):
            mp3_file = io.BytesIO(mp3_file)

        wav_output = io.BytesIO()
        cmd = "ffmpeg -hide_banner -loglevel panic -y -i pipe:0 -f wav pipe:1"

        # subprocess로 ffmpeg를 사용하여 MP3를 WAV로 변환
        process = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        wav_data, _ = process.communicate(input=mp3_file.read())

        wav_output.write(wav_data)
        wav_output.seek(0)  # BytesIO의 포인터를 처음으로 이동
        return wav_output

    def pcm_convert(self, wav_file):
        """Converts WAV (BytesIO) to PCM (NumPy array)."""
        wav_file.seek(0)  # BytesIO의 포인터를 처음으로 이동
        wav_data, sr = librosa.load(wav_file, sr=self.sr, mono=True)

        # PCM 데이터를 NumPy array 형태로 반환
        return wav_data
        