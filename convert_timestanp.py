import re
import sys
import os

def correct_timestamp(timestamp, assumed_fps=60.0, true_fps=59.94):
    """
    hh:mm:ss,ms または hh;mm;ss,ms を SRT形式（hh:mm:ss,mmm）に補正して返す。
    """
    parts = re.split(r'[:;]', timestamp.replace(',', ':'))
    hh, mm, ss, ms = map(int, parts)

    # 誤ったfpsに基づいた「見かけ上の秒数」
    wrong_total_seconds = hh * 3600 + mm * 60 + ss + (ms / 1000.0)

    # 正しい秒数に補正
    corrected_seconds = wrong_total_seconds * (assumed_fps / true_fps)

    # 秒数を hh:mm:ss,mmm に戻す
    corrected_hh = int(corrected_seconds // 3600)
    corrected_mm = int((corrected_seconds % 3600) // 60)
    corrected_ss = int(corrected_seconds % 60)
    corrected_ms = int((corrected_seconds - int(corrected_seconds)) * 1000)

    return f"{corrected_hh:02}:{corrected_mm:02}:{corrected_ss:02},{corrected_ms:03}"


def correct_offset_seconds(offset_seconds, assumed_fps=60.0, true_fps=59.94):
    """
    assumed_fps で計算された offset_seconds を、
    true_fps に基づいて補正する。

    例: 3600秒（1時間）を補正 → 約3603.6秒になる
    """
    return offset_seconds * (assumed_fps / true_fps)

def subtract_offset(timestamp_str, offset_seconds):
    """SRT形式の hh:mm:ss,mmm 文字列から秒数を引く"""
    hh, mm, ss_ms = timestamp_str.split(':')
    ss, ms = ss_ms.split(',')

    total_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
    adjusted = max(0.0, total_seconds - offset_seconds)

    hh = int(adjusted // 3600)
    mm = int((adjusted % 3600) // 60)
    ss = int(adjusted % 60)
    ms = int((adjusted - int(adjusted)) * 1000)

    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"

def process_file(file_path):
    # ファイル名と拡張子を分離
    base, ext = os.path.splitext(file_path)
    output_file = f"{base}_converted{ext}"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # タイムスタンプ検出パターン
    pattern = r"\b\d{2}[:;]\d{2}[:;]\d{2},\d{3}\b"
    timestamps = re.findall(pattern, content)

    if not timestamps:
        print("タイムスタンプが見つかりませんでした。")
        return

    # 最初のタイムスタンプを補正前に取得
    first_ts = timestamps[0]
    hh_part = int(re.split(r'[:;]', first_ts)[0])

    # hhが00以外の場合、その分の秒数をオフセットとして引く
    offset_seconds = hh_part * 3600 if hh_part != 0 else 0
    offset_seconds = correct_offset_seconds(offset_seconds)

    def replace_and_adjust(m):
        original = m.group()
        # corrected = correct_timestamp(original)
        corrected = original
        if offset_seconds:
            corrected = subtract_offset(corrected, offset_seconds)
        return corrected

    # タイムスタンプ置換
    converted_content = re.sub(pattern, replace_and_adjust, content)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(converted_content)

    print(f"変換後のファイルを保存しました: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python script.py 入力ファイル名")
    else:
        process_file(sys.argv[1])
