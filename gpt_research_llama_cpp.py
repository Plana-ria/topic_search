from llama_cpp import Llama
import os
import glob

# モデルのロード（ggufファイルのパスを指定）
model_path = "./models/DeepSeek-R1-Distill-Qwen-14B-Japanese-Q8_0.gguf"  # 量子化済モデルファイル（例）
llm = Llama(model_path=model_path, 
            n_gpu_layers=-1,
            n_ctx=1024*68)

def chunk_determining(chunk_text):
    prompt_content = (
        "以下の以下のインタビュー記録から、ユーザが求める発言や話題が含まれる箇所をすべて抜き出してください。\n"
        "抜き出した話題は話題の開始時間を明記してください。\n"
        "\n\n"
        "【ユーザの指示】\n"
        "地域の文化資源の活用に関する話題はありますか？"
        "【インタビューの文字起こしデータ】\n"
        f"{chunk_text}\n"
    )

    messages = [
        {"role": "user", "content": prompt_content}
    ]

    while True:
        response = llm.create_chat_completion(messages)
        text = response["choices"][0]["message"]["content"]
        print(text)
        exit(0)
        ans = extract_tagged_text(text, "<chunk>", "</chunk>")
        if ans and ans in ["True", "False"]:
            return ans == "True"
        print("再生成を試みます...")

def extract_tagged_text(text, start_tag, end_tag):
    start_index = text.find(start_tag)
    end_index = text.find(end_tag)
    if start_index != -1 and end_index != -1:
        return text[start_index + len(start_tag):end_index].strip()
    return None

def read_text_file_by_blank_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    blocks = [block.strip() for block in content.split('\n\n') if block.strip()]
    return blocks

def main():
    input_dir = "./data/sample"
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        chunk_determining(full_text)
    """
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_dir = "./data/dst"
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{file_name}_chunked.txt")
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + "\n-----\n")
        print(f"チャンクを保存しました: {output_file_path}")
    print("全てのファイルの処理が完了しました。")
    """


import time
if __name__ == "__main__":
    start_time = time.time()  # 開始時間を記録
    main()
    end_time = time.time()  # 終了時間を記録
    execution_time = end_time - start_time  # 実行時間を計算
    print(f"実行時間: {execution_time:.4f}秒")