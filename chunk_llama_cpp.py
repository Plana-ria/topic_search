from llama_cpp import Llama
import os
import glob

# モデルのロード（ggufファイルのパスを指定）
model_path = "./models/DeepSeek-R1-Distill-Qwen-14B-Japanese-Q8_0.gguf"  # 量子化済モデルファイル（例）
llm = Llama(model_path=model_path, n_ctx=4096)

def chunk_determining(latest_chunk, new_chunk, ahead_blocks):
    # チャット形式のメッセージリストを作成
    prompt_content = (
        "インタビューの文字起こしデータを文脈に沿ってチャンクに分けます。\n"
        "既存チャンクと新規チャンクが同一のチャンクとして結合すべきか、異なるチャンクとして分割すべきかを判断してください。\n"
        "参考のために新規チャンクの後に続く会話も提示しますので、新規チャンクの文脈を理解する参考にしてください。\n"
        "まず、「基本ルール」と「同じ話題と考える基準」そして「その他のルール」に従って結合か分割すべきかを考えます。\n"
        "次に、分割すべきと考えた場合は、「チャンク分割のルール」の要件を満たしているかを確認し、本当に分割するべきかの最終判断をしてください。\n"
        "【基本ルール】\n"
        "- 既存チャンクに含まれる発言数が5個未満の場合は、新規チャンクで明確かつ劇的な話題の転換が行われていない限り結合してください。\n"
        "- 既存チャンクに含まれる発言数が5個以上の場合も、同じ話題が続いている場合は結合してください。\n"
        "- 会話の流れが途切れる、または全く異なるテーマ（例：全く別のイベントや人物、背景、話題）が導入される場合のみ話題の転換とみなして分割します。(分割の判断をする前に、必ずチャンク分割のルールのチェック項目を確認してください)\n"
        "【同じ話題と考える基準】\n"
        "- 既存チャンクと新規チャンクが、同一のエピソードや話題の流れを補足・展開するものであれば、同じ話題と判断してください。\n"
        "- 話題が『補足』や『進展』として付け加えられている場合は、同じ話題と判断してください。\n"
        "- もし新規チャンクが既存チャンクの内容に対する質問や、関連する詳細を尋ねている場合、それは話題の継続として扱います。\n"
        "- 既存チャンクの文章と新規チャンクの文章をつなげた場合に、１つの文章として成立する場合は同じ話題と判断してください。\n"
        "【チャンク分割のルール】\n"
        "- 以下全ての項目に該当しない場合のみチャンクを分割してください。\n"
        #"-- もし新規チャンクを分割した時に、指示語や代名詞や文脈を踏まえた解釈が必要となり、以前のチャンク内容を参照しなければコンテキストがわからなくなる場合は、既存チャンクに結合してください。\n"
        "-- 同一人物の発言が連続している場合は、明確な話題の転換でない限りは結合してください。(話者名が記載されていない場合は直前の話者と同じ人物の発言と考えます)\n"
        "--. 既存チャンクが何の話題について話しているかが、ある程度明確にわかるまで結合します。既存チャンクが断片的すぎて話題がわからない状態の時は新規チャンクを結合してください。\n"
        "【その他のルール】\n"
        "- 相槌などは同じ話題が継続しているものと考えます。\n"
        "- 発言内容が存在しないチャンクは話題が継続していると考えます。\n"
        "- 判断が難しい場合は原則として話題が継続していると考えてください。\n"
        "---\n"
        "【既存チャンク】\n"
        f"{latest_chunk}\n"
        "---\n"
        "【新規チャンク】\n"
        f"{new_chunk}\n"
        "---\n"
        f"【後に続く会話】\n"
        f"{' '.join(ahead_blocks)}\n"
        "---\n"
        "最終的な回答は、結合する場合は<chunk>結合</chunk>、分割する場合は<chunk>分割</chunk>とタグで囲んでください。"
        "タグ内には結合か分割かのみを記述してください。"
    )

    messages = [
        {"role": "user", "content": prompt_content}
    ]

    while True:
        response = llm.create_chat_completion(messages)
        text = response["choices"][0]["message"]["content"]
        print(text)
        ans = extract_tagged_text(text, "<chunk>", "</chunk>")
        # すべての質問が抽出された場合、ループを終了
        if ans and ans in ["結合", "分割"]:
            if ans == "結合":
                return True
            elif ans == "分割":
                return False
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



def get_look_ahead_blocks(blocks, block_index, max_chars=1000):
    """
    現在のブロックの後ろのブロックを参照する際に、合計文字数がmax_charsを超えない範囲で取得する関数。
    """
    ahead_blocks = []
    total_length = 0
    for i in range(block_index, len(blocks)):
        block_length = len(blocks[i])
        total_length += block_length
        if total_length > max_chars:
            break
        ahead_blocks.append(blocks[i])
    return ahead_blocks





def main():
    # 入力と出力のディレクトリ
    input_dir = "./data/src"
    output_dir = "./data/dst"
    
    # 入力ディレクトリ内の全てのtxtファイルを取得
    txt_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    
    # ファイル数
    total_files = len(txt_files)

    for file_index, file_path in enumerate(txt_files, start=1):
        # 出力ファイルのパスを決定
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = os.path.join(output_dir, f"{file_name}_chunked.txt")
        
        # 出力ファイルが既に存在する場合はスキップ
        if os.path.exists(output_file_path):
            print(f"{output_file_path} はすでに処理済みです。")
            continue
        
        # ファイルを処理
        blocks = read_text_file_by_blank_lines(file_path)
        chunks = []
        latest_chunk = blocks[0]
        for block_index, new_chunk in enumerate(blocks[1:], start=2):  # 1つ目のブロックは既にlatest_chunkにセット
            # ブロックごとの進捗表示
            print(f"進捗: {block_index}/{len(blocks)} ブロック ({file_index}/{total_files} ファイル: {file_name}) ")
            
            # 現在のブロックの後ろ10個のブロック/1000文字未満を参照するための処理
            look_ahead_blocks = get_look_ahead_blocks(blocks, block_index)
            
            if chunk_determining(latest_chunk, new_chunk, look_ahead_blocks):
                latest_chunk += "\n\n" + new_chunk
            else:
                chunks.append(latest_chunk)
                latest_chunk = new_chunk

        # 最後のチャンクを追加
        chunks.append(latest_chunk)

        # チャンクを保存
        os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(chunk + "\n-----\n")
        
        print(f"チャンクを保存しました: {output_file_path}")

    print("全てのファイルの処理が完了しました。")



if __name__ == "__main__":
    main()
