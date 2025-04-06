import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from janome.tokenizer import Tokenizer
from sklearn.preprocessing import MinMaxScaler

# サラシナモデルをロード
model = SentenceTransformer("sbintuitions/sarashina-embedding-v1-1b")
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Janomeトークナイザーのインスタンス化
tokenizer = Tokenizer()

# 前処理済みデータの保存先パス
processed_data_path = "./tmp/processed_data.pkl"

# 事前に保存されたデータを読み込む関数
def load_processed_data():
    if os.path.exists(processed_data_path):
        with open(processed_data_path, "rb") as f:
            return pickle.load(f)
    return pd.DataFrame(columns=["Filename", "Utterance", "Embedding"])

# 保存された前処理データがあれば読み込む
processed_data = load_processed_data()

# 前処理の実行
def preprocess_data(files):
    global processed_data
    new_rows = []

    for file in files:
        filename = os.path.basename(file.name)
        with open(file.name, "r", encoding="utf-8") as f:
            content = f.read()
            utterances = [seg.strip() for seg in content.split("-----") if seg.strip()]
            for utterance in utterances:
                new_rows.append({"Filename": filename.split('_chunked')[0], "Utterance": utterance})

    new_data = pd.DataFrame(new_rows)

    # テキストのベクトル化
    embeddings = model.encode(new_data["Utterance"].tolist(), device=device)
    new_data["Embedding"] = embeddings.tolist()

    # データを統合・重複削除
    processed_data = pd.concat([processed_data, new_data], ignore_index=True)
    processed_data = processed_data.drop_duplicates(subset=["Utterance"])

    # 保存
    with open(processed_data_path, "wb") as f:
        pickle.dump(processed_data, f)

    return processed_data[["Filename", "Utterance", "Embedding"]]


# Janomeを使ったトークン化関数
def tokenize(text):
    return [token.surface for token in tokenizer.tokenize(text)]

# BM25の計算を行う関数
def calculate_bm25(query, bm25):
    query_tokens = tokenize(query)  # クエリをJanomeでトークン化
    scores = bm25.get_scores(query_tokens)
    return scores

# スコアの正規化を行う関数
def normalize_scores(scores):
    scaler = MinMaxScaler()
    return scaler.fit_transform(scores.reshape(-1, 1)).flatten()

# 検索関数
def search(query, use_bm25, use_vector, bm25_weight, vector_weight):
    # 事前に保存された文書のベクトルとBM25を読み込む
    global processed_data
    corpus = [tokenize(doc) for doc in processed_data["Utterance"].tolist()]  # Janomeでトークン化
    doc_embeddings = np.array(processed_data["Embedding"].tolist())
    bm25 = BM25Okapi(corpus)

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
        "Filename": processed_data["Filename"],
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
    df["Utterance"] = [utterance.replace('\n', '<br>') for utterance in df["Utterance"]]

    # 列名を変更
    df.columns = ["ファイル名", "内容", "BM25スコア", "ベクトルスコア", "最終スコア"]
    # HTMLに変換して返す
    return df.to_html(escape=False)
    #return (df[["Filename", "Utterance", "BM25 Score", "Vector Score", "Final Score"]]).to_html(escape=False)

def download_results(df):
    return df.to_csv(index=False).encode('utf-8')

def gpt_research(query):
    return f"GPTのリサーチ結果: {query} に関する情報"  # ダミー出力





with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.Tab("検索"):
            query = gr.Textbox(label="検索クエリ")
            use_bm25 = gr.Checkbox(label="BM25検索を使用", value=True)
            use_vector = gr.Checkbox(label="ベクトル検索を使用", value=True)
            bm25_weight = gr.Slider(0, 1, value=0.25, label="BM25の重み")
            vector_weight = gr.Slider(0, 1, value=0.75, label="ベクトル検索の重み")
            search_button = gr.Button("検索")
            result_table = gr.HTML()
            search_button.click(search, [query, use_bm25, use_vector, bm25_weight, vector_weight], result_table)
        
        with gr.Tab("GPTリサーチ"):
            gpt_query = gr.Textbox(label="調査クエリ")
            gpt_button = gr.Button("リサーチ")
            gpt_output = gr.Textbox(label="リサーチ結果")
            gpt_button.click(gpt_research, gpt_query, gpt_output)

        with gr.Tab("前処理"):
            file_input = gr.File(label="テキストファイルアップロード", file_types=[".txt"], file_count="multiple")
            preprocess_button = gr.Button("前処理実行")
            processed_data_output = gr.DataFrame(headers=["Filename", "Utterance", "Embedding"])
            preprocess_button.click(preprocess_data, file_input, processed_data_output)



demo.launch()
