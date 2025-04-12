from transformers import pipeline
from moviepy.editor import VideoFileClip
import torch
from datetime import timedelta
import sys
import os
from pydub.utils import mediainfo
import re
from pyannote.audio import Pipeline
from dotenv import load_dotenv
load_dotenv()

# MPS/CUDAの設定
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def replace_words(text):
    """
    単語置換ルールに基づいてテキストを変換する関数。
    """
    rules = {
        "反戦": "ハンセン",
        "反省": "ハンセン",
        "向井県": "裳掛",
        "愛戦": "愛生園",
        "愛声援": "愛生園",
        "蒸し明け": "虫明",
        "改心寮": "回春寮",
    }
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, rules.keys())) + r')\b')
    replaced_text = pattern.sub(lambda match: rules[match.group(0)], text)
    return replaced_text

def extract_audio(input_file, audio_file="./dara/tmp/temp_audio.wav"):
    """
    入力ファイルが動画の場合は音声を抽出する。  
    既に音声ファイルの場合はそのままのパスを返す。
    """
    file_info = mediainfo(input_file)
    if "audio" in file_info.get("codec_type", "") or input_file.lower().endswith((".wav", ".mp3", ".flac")):
        return input_file
    else:
        clip = VideoFileClip(input_file)
        clip.audio.write_audiofile(audio_file)
        clip.audio.close()
        clip.close()
        return audio_file

def transcribe_and_diarize_audio_file(audio_file, output_srt, num_speakers=1, language="japanese", batch_size=2, model_id="openai/whisper-large-v3-turbo"):
    """
    音声ファイルに対して以下を実施する:
      1. pyannote.audio による話者推定（指定した話者数）
      2. Whisperによるワード単位のトランスクリプション
      3. 各ワードに話者ラベルを割り当て、連続する同一話者のワードをグループ化しSRTファイルに出力
    """
    # --- ① pyannote.audio による話者推定 ---
    print("話者推定を実施...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                    use_auth_token=os.getenv("AUTH_TOKEN"),
                                    ).to(torch.device(device))
    # 指定した話者数で話者推定を実施
    kwargs = {}
    if num_speakers != 0:
        kwargs["num_speakers"] = int(num_speakers)
    diarization = diarization_pipeline(audio_file, **kwargs)
    
    # 指定時刻の話者を取得するヘルパー関数
    def get_speaker(time):
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.start <= time < segment.end:
                return speaker
        return "Unknown"

    # --- ② Whisperによるワード単位トランスクリプション ---
    print("Whisperによるトランスクリプションを実施...")
    # Whisperの音声認識パイプライン（word単位のタイムスタンプを返す）
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        batch_size=batch_size,
        device=device,
        use_fast=True
    )
    # generate_kwargsで"language"等を指定（必要に応じてパラメータ調整）
    kwargs = {"return_timestamps": "word", "generate_kwargs":{}}
    if language != "auto":
        kwargs["generate_kwargs"] = {"language": language}
    results = pipe(audio_file, **kwargs)
    results = results["chunks"]
    # 各チャンク内の単語（ワード）情報を1つのリストに統合
    words_list = []
    for i, chunk in enumerate(results):
        offset = 0 #float(pipe._preprocess_params["chunk_length_s"]) * i
        word = chunk["text"]
        start, end = chunk["timestamp"]  # タイムスタンプは (開始, 終了) のタプル
        words_list.append({"word": word, "start": start+offset, "end": end+offset})
    # 開始時刻順にソート
    words_list.sort(key=lambda x: x["start"])

    # --- ③ 単語毎の話者割当と連続する同一話者ワードのグループ化 ---
    print("同一話者ワードのグループ化を実施...")
    segments = []
    if words_list:
        # 先頭の単語で初期セグメントを作成
        current_segment = {
            "speaker": get_speaker(words_list[0]["start"]),
            "start": words_list[0]["start"],
            "end": words_list[0]["end"],
            "text": words_list[0]["word"]
        }
        for word in words_list[1:]:
            spk = get_speaker(word["start"])
            if re.search(r'[。！？?!]', current_segment["text"][-1]):
                segments.append(current_segment)
                current_segment = {
                    "speaker": spk,
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"]
                }
            elif spk == current_segment["speaker"] or spk == "Unknown" or current_segment["speaker"] == "Unknown":
                if current_segment["speaker"] == "Unknown" and spk != "Unknown":
                    current_segment["speaker"] = spk
                # 現在のセグメントにワードを追加
                current_segment["end"] = word["end"]
                current_segment["text"] += word["word"]
            else:
                # 話者が変わったらセグメントを確定
                segments.append(current_segment)
                current_segment = {
                    "speaker": spk,
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["word"]
                }
        segments.append(current_segment)

    # --- ④ SRTファイル作成 ---
    print("SRTファイルを作成...")
    with open(output_srt, "w", encoding="utf-8") as f:
        count = 1
        for seg in segments:
            # タイムスタンプ形式：HH:MM:SS,mmm
            start_time_str = str(timedelta(seconds=int(seg["start"]))) + f",{int((seg['start'] % 1) * 1000):03}"
            end_time_str = str(timedelta(seconds=int(seg["end"]))) + f",{int((seg['end'] % 1) * 1000):03}"
            # ルールに基づく単語変換
            text = replace_words(seg["text"])
            speaker_label = seg["speaker"]
            f.write(f"{count}\n")
            f.write(f"{start_time_str} --> {end_time_str}\n")
            # f.write(f"{speaker_label}: {text}\n\n")
            f.write(f"{speaker_label}\n")
            f.write(f"{text}\n\n")
            count += 1

def transcribe_file(input_file, output_srt, num_speakers=1, language="japanese", batch_size=2, model_id="openai/whisper-large-v3-turbo"):
    """
    入力ファイルのタイプに応じて適切に音声を抽出し、トランスクリプション＋話者推定を実施する
    """
    audio_file = extract_audio(input_file)
    transcribe_and_diarize_audio_file(audio_file, output_srt, num_speakers, language, batch_size, model_id)
    # 一時的な音声ファイルが生成されている場合は削除
    if audio_file == "./dara/tmp/temp_audio.wav" and os.path.exists(audio_file):
        os.remove(audio_file)
    return output_srt, "文字起こしが完了しました"

def main():
    if len(sys.argv) < 3:
        print("使い方: python script.py 入力ファイルのパス 話者数")
        sys.exit(1)
    input_file = sys.argv[1]
    num_speakers = sys.argv[2]
    output_srt = os.path.join(os.path.dirname(os.path.abspath(input_file)),
                              f"{os.path.splitext(os.path.basename(input_file))[0]}.srt")
    transcribe_file(input_file, output_srt, num_speakers)

if __name__ == "__main__":
    main()
