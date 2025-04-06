from transformers import AutoTokenizer

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained("cyberagent/DeepSeek-R1-Distill-Qwen-14B-Japanese")

# 対象のファイルパス（必要に応じて書き換えてください）
file_path = "./data/srt/250313_大西_文字起こし.txt"

# ファイルを読み込み、トークン数を計算
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# トークンIDに変換
token_ids = tokenizer.encode(text, add_special_tokens=False)

# トークン数表示
print(f"ファイル: {file_path}")
print(f"トークン数: {len(token_ids)}")
