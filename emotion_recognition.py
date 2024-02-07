import time
import joblib
import numpy as np

from pylsl import StreamInlet, resolve_stream
from sklearn.preprocessing import normalize
from fft_process import fft_process


def predict(data, model):
    output = model.predict(data)
    return np.mean(output)


def emotion_processing(queue):
    # 加载情绪识别模型
    Val_R = joblib.load("D:/eeg-model/val_model.pkl")
    Aro_R = joblib.load("D:/eeg-model/aro_model.pkl")

    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])
    print("Connect success! Working on...")

    tmp = []
    while True:
        sample, timestamp = inlet.pull_sample()
        tmp.append(sample)
        # 每收集到30s的EEG信号做一次情绪识别
        if len(tmp) >= 250 * 30:
            data = np.array(tmp)
            data = data.T  # channels * samples
            # fft处理
            fft_data = fft_process(data)
            fft_data = normalize(fft_data)

            # start prediction
            score_valence = predict(fft_data, Val_R)
            score_arousal = predict(fft_data, Aro_R)

            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            result = {
                'time': current_time,
                'valence': score_valence,
                'arousal': score_arousal
            }
            queue.put(result)

            tmp.clear()
