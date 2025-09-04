import { useState, useEffect } from "react";

export default function Home() {
  const [message, setMessage] = useState("");
  const [recognizedText, setRecognizedText] = useState("");
  const [command, setCommand] = useState("");

  useEffect(() => {
    fetch("http://localhost:5000/")
      .then((res) => res.text())
      .then((data) => setMessage(data));
  }, []);

  // 録音開始・送信処理
  const handleRecord = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);

    let chunks = [];
    mediaRecorder.ondataavailable = (e) => chunks.push(e.data);

    mediaRecorder.onstop = async () => {
  const blob = new Blob(chunks, { type: "audio/webm" });

  // ---- WAV に変換する処理 ----
  const arrayBuffer = await blob.arrayBuffer();
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

  // PCM 16bit little-endian に変換
  const wavBuffer = audioBufferToWav(audioBuffer);
  const wavBlob = new Blob([wavBuffer], { type: "audio/wav" });

  // Flask に送信
  const formData = new FormData();
  formData.append("file", wavBlob, "recording.wav");

  const res = await fetch("http://localhost:5000/recognize", {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  console.log("Flaskからの応答:", data);   
  setRecognizedText(data.result || "認識できませんでした");
  setCommand(data.command || "a");
};

    mediaRecorder.start();
    setTimeout(() => mediaRecorder.stop(), 3000); // 3秒録音
  };

  // Web Audio → WAV 変換関数
  function audioBufferToWav(buffer) {
    const numOfChan = buffer.numberOfChannels;
    const length = buffer.length * numOfChan * 2 + 44;
    const bufferArray = new ArrayBuffer(length);
    const view = new DataView(bufferArray);

    // WAV ヘッダ書き込み
    writeUTFBytes(view, 0, "RIFF");
    view.setUint32(4, 36 + buffer.length * numOfChan * 2, true);
    writeUTFBytes(view, 8, "WAVE");
    writeUTFBytes(view, 12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numOfChan, true);
    view.setUint32(24, buffer.sampleRate, true);
    view.setUint32(28, buffer.sampleRate * numOfChan * 2, true);
    view.setUint16(32, numOfChan * 2, true);
    view.setUint16(34, 16, true);
    writeUTFBytes(view, 36, "data");
    view.setUint32(40, buffer.length * numOfChan * 2, true);

    // PCM データ書き込み
    let offset = 44;
    const interleaved = interleave(buffer);
    for (let i = 0; i < interleaved.length; i++, offset += 2) {
      const s = Math.max(-1, Math.min(1, interleaved[i]));
      view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }

    return view.buffer;
  }

  function interleave(input) {
    if (input.numberOfChannels === 2) {
      const left = input.getChannelData(0);
      const right = input.getChannelData(1);
      const interleaved = new Float32Array(left.length + right.length);
      let index = 0;
      let inputIndex = 0;
      while (index < interleaved.length) {
        interleaved[index++] = left[inputIndex];
        interleaved[index++] = right[inputIndex];
        inputIndex++;
      }
      return interleaved;
    } else {
      return input.getChannelData(0);
    }
  }

  function writeUTFBytes(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  return (
    <div className="flex flex-col items-center justify-center h-screen">
      <h1 className="text-xl font-bold">Tello音声認識</h1>
      <p>{message}</p>
      <button
        onClick={handleRecord}
        className="bg-blue-500 text-white px-4 py-2 rounded mt-4"
      >
        🎤 録音開始
      </button>
      <div className="mt-4">
        <h2>送信するコマンド：</h2>
        <p>{command}</p>
      </div>
    </div>
  );
}