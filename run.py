import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import gradio as gr
import pandas as pd
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from janome.tokenizer import Tokenizer
from sklearn.preprocessing import MinMaxScaler
from transcribe import transcribe_file

# モデルのロード
model = SentenceTransformer("sbintuitions/sarashina-embedding-v1-1b")
tokenizer = Tokenizer()

# 前処理済みデータの保存先パス
processed_data_dir = "./data/tmp"
# ./data/tmp ディレクトリが存在しない場合は作成
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)




# 保存された前処理データを読み込む関数
def load_processed_data(dir):
    processed_data = pd.DataFrame(columns=["id", "Utterance", "Embedding"])
    # ./data/tmp ディレクトリ内の全てのファイルを読み込む
    for file_name in os.listdir(f"{processed_data_dir}/{dir}"):
        if file_name.endswith(".pkl"):  # pklファイルのみ読み込む
            file_path = os.path.join(f"{processed_data_dir}/{dir}", file_name)
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                processed_data = pd.concat([processed_data, data], ignore_index=True)
    
    return processed_data

# 前処理を実行する関数
def preprocess_data(files, separators=["-----","\n\n"], dir_name="gpt_chunk"):
    flag = False
    for file_path in files:
        new_rows = []  # ループごとにnew_rowsを初期化する
        filename = os.path.basename(file_path).split('_chunked')[0]  # '_chunked'を取り除く
        output_dir = f"{processed_data_dir}/{dir_name}"
        output_path = f"{output_dir}/{filename}.pkl"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # すでに処理されたファイルが存在する場合はスキップ
        if os.path.exists(output_path):
            print(f"ファイル {filename} はすでに処理されています。")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # 分割処理
            pattern = "|".join(map(re.escape, separators))
            utterances = [seg.strip() for seg in re.split(pattern, content) if seg.strip()]
            max_id_length = len(str(len(utterances)))  # 最大桁数を取得
            for id, utterance in enumerate(utterances):
                new_rows.append({"id": f"{filename}_{str(id).zfill( max_id_length+1 )}", "Utterance": utterance})

        new_data = pd.DataFrame(new_rows)

        # テキストのベクトル化
        embeddings = model.encode(new_data["Utterance"].tolist())
        new_data["Embedding"] = embeddings.tolist()


        # 保存
        with open(output_path, "wb") as f:
            pickle.dump(new_data, f)
        flag = True

    # 結果を返す
    return flag




# アプリ起動時に前処理を実行する
def reprocess_data():
    global processed_gpt_chunk_data
    global processed_srt_chunk_data
    global bm25_gpt
    global bm25_srt
    # 前処理を実行
    txt_files = [file for file in os.listdir("./data/txt") if file.endswith(".txt")]
    txt_file_paths = [os.path.join("./data/txt", file) for file in txt_files]
    flag = preprocess_data(txt_file_paths, ["-----"], "gpt_chunk")
    processed_gpt_chunk_data = load_processed_data("gpt_chunk")
    if flag or (not os.path.exists("./data/tmp/bm25/bm25_gpt.pkl")):
        load_bm25_index(processed_gpt_chunk_data, "gpt")
    with open("./data/tmp/bm25/bm25_gpt.pkl", 'rb') as file:
        bm25_gpt = pickle.load(file)
    txt_files = [file for file in os.listdir("./data/srt") if file.endswith(".txt")]
    txt_file_paths = [os.path.join("./data/srt", file) for file in txt_files]
    flag = preprocess_data(txt_file_paths, ["-----", "\n\n"], "srt_chunk")
    processed_srt_chunk_data = load_processed_data("srt_chunk")
    if flag or (not os.path.exists("./data/tmp/bm25/bm25_srt.pkl")):
        load_bm25_index(processed_srt_chunk_data, "srt")
    with open("./data/tmp/bm25/bm25_srt.pkl", 'rb') as file:
        bm25_srt = pickle.load(file)
    











# Janomeを使ったトークン化関数
def tokenize(text):
    return [token.surface for token in tokenizer.tokenize(text)]

def safe_bm25_scores(query_tokens, bm25):
    num_docs = len(bm25.doc_freqs)  # corpusの代わりにdoc_freqsから文書数を取得
    scores = np.zeros(num_docs)
    for token in query_tokens:
        if token in bm25.idf:  # 語がコーパスに存在するときだけ加算
            token_scores = bm25.get_scores([token])
            scores += token_scores
    return scores


# BM25の計算を行う関数
def calculate_bm25(query, bm25):
    query_tokens = tokenize(query)
    return safe_bm25_scores(query_tokens, bm25)


def create_bm25_index(processed_data, pkl_path):
    corpus = [tokenize(doc) for doc in processed_data["Utterance"].tolist()]  # Janomeでトークン化
    bm25 = BM25Okapi(corpus)
    # 保存
    with open(pkl_path, "wb") as f:
        pickle.dump(bm25, f)

def load_bm25_index(processed_data, name):
    bm25_dir = "./data/tmp/bm25"
    if not os.path.exists(bm25_dir):
        os.makedirs(bm25_dir)
    create_bm25_index(processed_data, f"{bm25_dir}/bm25_{name}.pkl")

# スコアの正規化を行う関数
def normalize_scores(scores):
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()


# 検索関数
def search(query, use_bm25, use_vector, bm25_weight, vector_weight, processed_data, bm25):
    bm25_weight = float(bm25_weight)
    vector_weight = float(vector_weight)
    doc_embeddings = np.array(processed_data["Embedding"].tolist())
    # ベクトル検索用のクエリの埋め込み計算
    query_embedding = model.encode([query])[0]

    # BM25スコアの計算
    bm25_scores = calculate_bm25(query, bm25) if use_bm25 else np.zeros(len(processed_data))
    
    # ベクトル検索のスコア計算
    vector_scores = np.array([np.dot(query_embedding, doc_embedding) for doc_embedding in doc_embeddings]) if use_vector else np.zeros(len(processed_data))
    
    # スコアを正規化
    bm25_scores_normalized = normalize_scores(bm25_scores)
    vector_scores_normalized = normalize_scores(vector_scores)

    # データフレームの作成
    data = {
        "id": processed_data["id"],
        "Utterance": processed_data["Utterance"],
        "BM25 Score": bm25_scores_normalized,
        "Vector Score": vector_scores_normalized,
    }
    df = pd.DataFrame(data)
    
    # スコアの重み付け
    if use_bm25 and use_vector:
        df["Final Score"] = df["BM25 Score"] * bm25_weight + df["Vector Score"] * vector_weight
    elif use_bm25:
        df["Final Score"] = df["BM25 Score"]
    elif use_vector:
        df["Final Score"] = df["Vector Score"]
    else:
        df["Final Score"] = 0
    
    # スコアのソート
    df = df.sort_values(by="Final Score", ascending=False)
    # スコアを小数点3桁に丸める
    df["BM25 Score"] = df["BM25 Score"].round(3)
    df["Vector Score"] = df["Vector Score"].round(3)
    df["Final Score"] = df["Final Score"].round(3)
    df["Utterance"] = df["Utterance"].apply(transform_utterance_text)
    

    # 列名を変更
    df.columns = ["ID", "内容", "キーワードスコア", "意味スコア", "合計スコア"]
    return df


def download_results(df):
    return df.to_csv(index=False).encode('utf-8')

def gpt_research(query):
    return f"GPTのリサーチ結果: {query} に関する情報"  # ダミー出力




def transform_utterance_text(utterance: str) -> str:
    """
    空白行でブロックに分割し、各ブロックの改行区切りテキストを
    最後の行を除いて【】で囲み、最後の行をその外に結合する。
    すべてのブロックを結合して1つの文字列として返す。
    """
    # 空白行で分割（連続する改行2つ以上）
    blocks = utterance.strip().split('\n\n')
    transformed_blocks = []

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) > 1:
            content = '【' + " | ".join(lines[:-1]) + '】' + lines[-1]
        else:
            content = lines[0]
        transformed_blocks.append(content)

    return '\n\n'.join(transformed_blocks)



def update_weights(weight):
    # bm25_weightとvector_weightの合計が1になるように設定
    bm25_weight = weight
    vector_weight = 1 - weight
    return bm25_weight, vector_weight


# 起動時に前処理を実行
processed_gpt_chunk_data = None
processed_srt_chunk_data = None
bm25_gpt = None
bm25_srt = None
reprocess_data()



# 出力ファイルの自動生成（入力ファイルが選ばれた時）
def set_output_path(input_path):
    if not input_path:
        return ""
    base, _ = os.path.splitext(input_path)
    return base + ".srt"

languages =[
    "auto",
    "javanese",
    "english",
    "chinese",
    "german",
    "spanish",
    "russian",
    "korean",
    "french",
    "japanese",
    "portuguese",
    "turkish",
    "polish",
    "catalan",
    "dutch",
    "arabic",
    "swedish",
    "italian",
    "indonesian",
    "hindi",
    "finnish",
    "vietnamese",
    "hebrew",
    "ukrainian",
    "greek",
    "malay",
    "czech",
    "romanian",
    "danish",
    "hungarian",
    "tamil",
    "norwegian",
    "thai",
    "urdu",
    "croatian",
    "bulgarian",
    "lithuanian",
    "latin",
    "maori",
    "malayalam",
    "welsh",
    "slovak",
    "telugu",
    "persian",
    "latvian",
    "bengali",
    "serbian",
    "azerbaijani",
    "slovenian",
    "kannada",
    "estonian",
    "macedonian",
    "breton",
    "basque",
    "icelandic",
    "armenian",
    "nepali",
    "mongolian",
    "bosnian",
    "kazakh",
    "albanian",
    "swahili",
    "galician",
    "marathi",
    "punjabi",
    "sinhala",
    "khmer",
    "shona",
    "yoruba",
    "somali",
    "afrikaans",
    "occitan",
    "georgian",
    "belarusian",
    "tajik",
    "sindhi",
    "gujarati",
    "amharic",
    "yiddish",
    "lao",
    "uzbek",
    "faroese",
    "haitian creole",
    "pashto",
    "turkmen",
    "nynorsk",
    "maltese",
    "sanskrit",
    "luxembourgish",
    "myanmar",
    "tibetan",
    "tagalog",
    "malagasy",
    "assamese",
    "tatar",
    "hawaiian",
    "lingala",
    "hausa",
    "bashkir",
    "sundanese",
    "cantonese"
]


with gr.Blocks() as demo:
    with gr.Tabs():

        with gr.Tab("検索（オリジナルの分割）"):
            query = gr.Textbox(label="検索クエリ")
            with gr.Accordion("検索のオプション", open=False):
                use_bm25 = gr.Checkbox(label="キーワード検索を使用", value=True)
                use_vector = gr.Checkbox(label="意味検索を使用", value=True)
                with gr.Group():
                    weight_slider = gr.Slider(0, 1, value=0.5, label="重みの割合", step=0.01)
                    with gr.Row():
                        bm25_weight = gr.Textbox(label="キーワード検索の重み", interactive=False, value=0.5)
                        vector_weight = gr.Textbox(label="意味検索の重み", interactive=False, value=0.5)
                # スライダーの値が変わると、bm25_weightとvector_weightが更新される
                weight_slider.change(fn=update_weights, inputs=weight_slider, outputs=[bm25_weight, vector_weight])

            search_button = gr.Button("検索")
            #result_table = gr.HTML(elem_id="scrollable-html")

            result_table = gr.DataFrame(
                headers=["ID", "内容", "キーワードスコア", "意味スコア", "最終スコア"],
                column_widths=["15%", "55%", "10%", "10%", "10%"],
                wrap=True
            )
            search_button.click(search, [query, use_bm25, use_vector, bm25_weight, vector_weight, gr.State(processed_srt_chunk_data), gr.State(bm25_srt)], result_table)

        with gr.Tab("検索（話題による分割）"):
            query = gr.Textbox(label="検索クエリ")
            with gr.Accordion("検索のオプション", open=False):
                use_bm25 = gr.Checkbox(label="キーワード検索を使用", value=True)
                use_vector = gr.Checkbox(label="意味検索を使用", value=True)
                with gr.Group():
                    weight_slider = gr.Slider(0, 1, value=0.5, label="重みの割合", step=0.01)
                    with gr.Row():
                        bm25_weight = gr.Textbox(label="キーワード検索の重み", interactive=False, value=0.5)
                        vector_weight = gr.Textbox(label="意味検索の重み", interactive=False, value=0.5)
                # スライダーの値が変わると、bm25_weightとvector_weightが更新される
                weight_slider.change(fn=update_weights, inputs=weight_slider, outputs=[bm25_weight, vector_weight])
            search_button = gr.Button("検索")
            #result_table = gr.HTML(elem_id="scrollable-html")

            result_table = gr.DataFrame(
                headers=["ID", "内容", "キーワードスコア", "意味スコア", "最終スコア"],
                column_widths=["15%", "55%", "10%", "10%", "10%"],
                wrap=True
            )

            search_button.click(search, [query, use_bm25, use_vector, bm25_weight, vector_weight, gr.State(processed_gpt_chunk_data), gr.State(bm25_gpt)], result_table)

        with gr.Tab("AI文字起こし"):
            input_file = gr.File(label="音声ファイルをドロップまたは選択", type="filepath")
            output_file_path = gr.Textbox(label="出力するSRTファイルのパス")
            num_speakers = gr.Slider(0, 10, value=0, step=1, label="最大話者数",
                                     info="0に設定すると自動的に話者数を推定します。実際の話者数より少ない場合は、話者推定がうまく実行されない場合があります。"
                                     )
            language = gr.Dropdown(choices=languages, value="japanese", label="言語")
            model_id = gr.Dropdown(choices=[
                "openai/whisper-large-v3",
                "openai/whisper-large-v3-turbo", 
                "openai/whisper-medium",
                "openai/whisper-small",
                "openai/whisper-base",
                "openai/whisper-tiny",
            ], value="openai/whisper-large-v3-turbo", label="モデル")
            batch_size = gr.Slider(1, 32, value=2, step=1, label="バッチサイズ")

            transcribe_button = gr.Button("文字起こしを実行")
            result_file = gr.File(label="出力されたSRTファイル")
            status_text = gr.Textbox(label="ステータス", interactive=False)

            # 入力ファイル変更時に出力パスを自動生成
            input_file.change(fn=set_output_path, inputs=input_file, outputs=output_file_path)

            # 実行ボタンで文字起こし
            transcribe_button.click(
                fn=transcribe_file,
                inputs=[input_file, output_file_path, num_speakers, language, batch_size, model_id],
                outputs=[result_file, status_text]
            )





demo.css = ".gradio-container { width: 90%; margin: auto; }"
demo.launch()
