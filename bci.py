import argparse
import time
import joblib
import numpy as np
import pyeeg as pe
from brainflow import DataFilter, WaveletTypes, WaveletDenoisingTypes, ThresholdTypes, WaveletExtensionTypes, \
    NoiseEstimationLevelTypes, FilterTypes, LogLevels
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowPresets
from sklearn.preprocessing import normalize
from EmotionCache import Node, cache


EEG_SEGMENT = 250*60  # 截取1分钟长度的eeg
band = [4, 8, 12, 30, 45]  # 4 bands
window_size = 500  # Averaging band power of 2 sec
step_size = 31  # Each 0.125 sec update once
eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
sfreq = BoardShim.get_sampling_rate(BoardIds.CYTON_BOARD.value)
# 加载情绪识别模型
model_valence = joblib.load("model/valence_model.pkl")
model_arousal = joblib.load("model/arousal_model.pkl")


def preprocess(data):
    """
    预处理
    :param data: 原始eeg信号
    :return: 预处理后的eeg信号
    """
    # 筛选通道
    eeg_data = data[eeg_channels, :]
    for idx, channel in enumerate(eeg_channels):
        # 带通滤波
        DataFilter.perform_bandpass(eeg_data[idx], sampling_rate=sfreq, start_freq=4., stop_freq=45.,
                                    filter_type=FilterTypes.BUTTERWORTH, order=4, ripple=0)
        # 小波去噪
        DataFilter.perform_wavelet_denoising(eeg_data[idx], WaveletTypes.BIOR3_9, 3,
                                             WaveletDenoisingTypes.SURESHRINK, ThresholdTypes.HARD,
                                             WaveletExtensionTypes.SYMMETRIC, NoiseEstimationLevelTypes.FIRST_LEVEL)
    return eeg_data


def extract_feature_psi(eeg_data):
    """
    提取eeg的psi特征
    :param eeg_data: eeg信号序列，[channels * datapoints]
    :return: 标准化后的特征列表
    """
    start = 0
    feature_list = []
    while start + window_size < eeg_data.shape[1]:
        psi_data = []
        for i in range(len(eeg_data)):
            x = eeg_data[i][start: start + window_size]  # Slice raw data over 2 sec, at interval of 0.125 sec
            y = pe.bin_power(x, band, sfreq)  # FFT over 2 sec of channel j
            psi_data = psi_data + list(y[0])
        feature_list.append(np.array(psi_data))
        start = start + step_size
    feature_list = np.array(feature_list)
    return normalize(feature_list)


def get_emotion(feature_list):
    """
    预测情绪
    :param feature_list: 特征序列
    :return: valence，arousal
    """
    valence = np.mean(model_valence.predict(feature_list))
    arousal = np.mean(model_arousal.predict(feature_list))

    return valence, arousal


BoardShim.set_log_level(LogLevels.LEVEL_ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                    default=0)
parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                    default=0)
parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='COM3')
parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                    required=False, default=BoardIds.CYTON_BOARD)
parser.add_argument('--file', type=str, help='file', required=False, default='')
parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                    required=False, default=BoardIds.NO_BOARD)
args = parser.parse_args()

params = BrainFlowInputParams()
params.ip_port = args.ip_port
params.serial_port = args.serial_port
params.mac_address = args.mac_address
params.other_info = args.other_info
params.serial_number = args.serial_number
params.ip_address = args.ip_address
params.ip_protocol = args.ip_protocol
params.timeout = args.timeout
params.file = args.file
params.master_board = args.master_board

# 开启eeg流式传输
board = BoardShim(args.board_id, params)
board.prepare_session()
board.start_stream()

try:
    while True:
        time.sleep(10)
        data = board.get_board_data(num_samples=EEG_SEGMENT)

        eeg_data = preprocess(data)
        feature_list = extract_feature_psi(eeg_data)
        valence, arousal = get_emotion(feature_list)
        cache.put(Node(valence, arousal))
        print(cache.queue)
finally:
    board.stop_stream()
    board.release_session()
