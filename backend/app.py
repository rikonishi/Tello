from flask import Flask, request, jsonify
import os
import io
from flask_cors import CORS

import numpy as np
import torch
import json
import wave

from my_model import MyCTCModel
import decode_function as df

import socket

TELLO_IP = '192.168.10.1'
TELLO_PORT = 8889
TELLO_ADDRESS = (TELLO_IP, TELLO_PORT)

# UDPソケット
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def edit_dist(a, b, add=1, remove=1, replace=1):
  len_a = len(a) + 1
  len_b = len(b) + 1
  # 配列の初期化
  arr = [[-1 for col in range(len_a)] for row in range(len_b)]
  arr[0][0] = 0
  for row in range(1, len_b):
    arr[row][0] = arr[row - 1][0] + add
  for col in range(1, len_a):
    arr[0][col] = arr[0][col - 1] + remove
  # 編集距離の計算
  def go(row, col):
    if (arr[row][col] != -1):
      return arr[row][col]
    else:
      dist1 = go(row - 1, col) + add
      dist2 = go(row, col - 1) + remove
      dist3 = go(row - 1, col - 1)
      arr[row][col] = min(dist1, dist2, dist3) if (b[row - 1] == a[col - 1]) else min(dist1, dist2, dist3 + replace)
      return arr[row][col]
  return go(len_b - 1, len_a - 1)

app = Flask(__name__)
CORS(app)


@app.route("/recognize", methods=["POST"])
def recognize_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]

    # そのままWAVとして読み込み
    audio_bytes = file.read()
    wav_io = io.BytesIO(audio_bytes)

    with wave.open(wav_io, "rb") as wav:
        fs = wav.getframerate()
        num_samples = wav.getnframes()
        waveform = wav.readframes(num_samples)
        waveform = np.frombuffer(waveform, dtype=np.int16)

    # === 特徴量抽出 ===
    feat_extractor = df.FeatureExtractor(sample_frequency=16000)
    mfcc = feat_extractor.ComputeMFCC(waveform)
    mfcc = mfcc.astype(np.float32)
    num_frames, feat_dim = mfcc.shape

    # === モデルと設定のロード ===
    exp_dir = "/Users/Owner/webアプリ開発/backend/ex"  
    unit = "kana"
    model_dir = os.path.join(exp_dir, unit + "_model_ctc")
    model_file = os.path.join(model_dir, "final_model.pt")
    mean_std_file = os.path.join(model_dir, "mean_std.txt")
    token_list_path = os.path.join(exp_dir, "data", unit, "token_list")
    config_file = os.path.join(model_dir, "config.json")

    # 設定ファイルをロード
    with open(config_file, "r") as f:
        config = json.load(f)

    # 平均・標準偏差
    with open(mean_std_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        feat_mean = np.array(lines[1].split(), dtype=np.float32)
        feat_std = np.array(lines[3].split(), dtype=np.float32)

    feat = (mfcc - feat_mean) / feat_std
    feat = torch.tensor([feat.tolist()])

    # モデル構築
    num_tokens = sum(1 for _ in open(token_list_path, encoding="utf-8")) + 1
    model = MyCTCModel(dim_in=feat_dim,
                       dim_enc_hid=config["hidden_dim"],
                       dim_enc_proj=config["projection_dim"],
                       dim_out=num_tokens,
                       enc_num_layers=config["num_layers"],
                       enc_bidirectional=config["bidirectional"],
                       enc_sub_sample=config["sub_sample"],
                       enc_rnn_type=config["rnn_type"])
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model.eval()

   # 推論
    feat_lens = torch.tensor([num_frames])  
    outputs, _ = model(feat, feat_lens)
    _, hyp_per_frame = torch.max(outputs[0], 1)
    hyp_per_frame = hyp_per_frame.cpu().numpy()




    # トークンリストを辞書に
    token_list = {0: "<blank>"}
    with open(token_list_path, "r", encoding="utf-8") as f:
        for line in f:
            token, idx = line.split()
            token_list[int(idx)] = token

    hypothesis = df.ctc_simple_decode(hyp_per_frame, token_list)
    string_hypothesis = "".join(hypothesis)

    print("音声認識結果"+string_hypothesis)

    # 近い単語を探す

    dictionary = ["じょーしょう","かこう","まえ","うしろ","みぎ","ひだり","りりく","ちゃくりく","すすむ","まがる","ひゃくせんちめーとる","にひゃくせんちめーとる","ゆっくり","はやく","ストリームオン","ストリームオフ","ちゅーがえり","に","して"]
    dictionary_phone = ["joushou","kakou","mae","usiro","migi","hidari","ririku","chakuriku","susumu","magaru","hyakusenchimeetoru","nihyakusentimeetoru","yukkuri","hayaku","sutoriimuon","sutoriimuofu","chuugaeri","ni","site"]
    dictionary_char = ["上昇","下降","前","後ろ","右","左","離陸","着陸","進む","曲がる","100cm","200cm","ゆっくり","早く","ストリームオン","ストリームオフ","宙返り","に","して"]
    dict = []
    distance_bet_wards = []
    time = []
    parts = []
    string = ""
    i = 0

    while i < len(string_hypothesis):
        for n in range(i,len(string_hypothesis)+1):
            for ward in dictionary:
                if edit_dist(string_hypothesis[i:n],ward)>edit_dist(string_hypothesis[i:n+1],ward):
                    time.append(n+1)
                    distance_bet_wards.append(edit_dist(string_hypothesis[i:n],ward))
                    dict.append(ward)
        if not distance_bet_wards:
            break
        else:
            min_index=distance_bet_wards.index(min(distance_bet_wards))
            i = time[min_index]
            index=dictionary.index(dict[min_index])
            string = string + dictionary_char[index]
            parts.append(dictionary[index])
            time = []
            distance_bet_wards = []
            dict = []


    print("レーベンシュタイン距離を測って得た文字列:")

    command = ""
    distance = ""
    angle = ""
    direction = ""

    for part in parts:
        if(command != "up" or command !="down"or command !="cw" or command!="land" or command!="takeoff"):
            if (direction == "f" or direction =="b"or direction =="l" or direction =="r"):
                if part == "ちゅーがえり":
                    command = "flip"
        if not command:
            if(part=="じょうしょう"):
                command = "up"
            elif(part=="かこう"):
                command = "down"
            elif(part == "まえ"):
                command = "forward"
                direction = "f"
            elif(part=="うしろ"):
                command = "back"
                direction = "b"
            elif(part=="みぎ"):
                command = "right"
                direction = "r"
            elif(part=="ひだり"):
                command = "left"
                direction = "l"
            elif(part=="曲がる"):
                command = "cw"
            elif(part =="ストリームオン"):
                command ="streamon"
            elif(part=="ストリームオフ"):
                command = "streamoff"
        if not distance:
            if(part == "ひゃくせんちめーとる"):
                distance = "100"
            elif(part == "ちゃくりく"):
                command = "land"
            elif(part == "りりく"):
                command = "takeoff" 

    if not command:
        sentence = "land"
    elif(command == "flip"):
        if not direction:
            direction = "f"
        sentence = command +" "+direction
    elif(command == "up" or command=="down"or command=="forward" or command== "back" or command=="right" or command == "left"):
        if not distance:
            distance = "100"
        sentence = command + " "+distance
    elif(command =="takeoff" or command == "land"):
        sentence = command

    try:
        sock.sendto(command.encode("utf-8"), TELLO_ADDRESS)
        print(f"✅ Telloに送信: {command}")
    except Exception as e:
        print(f"❌ Tello送信エラー: {e}")

    return jsonify({"result": string_hypothesis, "command": sentence})

@app.route("/")
def hello():
    return "pythonサーバーと接続しました"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)