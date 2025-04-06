import os
import csv

# ソースディレクトリとデスティネーションディレクトリ
src_dir = './data/src'
dst_dir = './data/dst'

# デスティネーションディレクトリが存在しない場合は作成
os.makedirs(dst_dir, exist_ok=True)

# CSVの出力ファイルパス
output_csv = os.path.join(dst_dir, 'output.csv')

# CSVに書き込むための準備
with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)

    # ソースディレクトリ内のファイルを処理
    for filename in os.listdir(src_dir):
        file_path = os.path.join(src_dir, filename)
        
        # ファイルがテキストファイルの場合のみ処理
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            # 拡張子を取り除いたファイル名を取得
            filename_without_ext = os.path.splitext(filename)[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # 空白行で分割して塊にする
                blocks = content.split('\n\n')
                
                # 各塊を|で改行置き換え
                processed_content = '\n'.join(block.replace('\n', ' | ') for block in blocks)

                # 各行（block）が空白や改行のみの場合は無視
                for block in processed_content.split('\n'):
                    if block.strip():  # 空白や改行のみの行を無視
                        writer.writerow([filename_without_ext, block])

print(f"CSVファイル '{output_csv}' に書き出しました。")
