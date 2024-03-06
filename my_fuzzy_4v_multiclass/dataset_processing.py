from scipy import signal

import matplotlib.pyplot as plt
import scipy.io as sio
import neurokit2 as nk
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time


def preprocess_EEG(raw, feature):
    overall = signal.firwin(9, [0.0625, 0.46875], window="hamming")
    theta = signal.firwin(9, [0.0625, 0.125], window="hamming")
    alpha = signal.firwin(9, [0.125, 0.203125], window="hamming")
    beta = signal.firwin(9, [0.203125, 0.46875], window="hamming")
    filt_data = signal.filtfilt(overall, 1, raw)
    filt_theta = signal.filtfilt(theta, 1, filt_data)
    filt_alpha = signal.filtfilt(alpha, 1, filt_data)
    filt_beta = signal.filtfilt(beta, 1, filt_data)
    ftheta, psdtheta = signal.welch(filt_theta, nperseg=256)
    falpha, psdalpha = signal.welch(filt_alpha, nperseg=256)
    fbeta, psdbeta = signal.welch(filt_beta, nperseg=256)
    feature.append(max(psdtheta))
    feature.append(max(psdalpha))
    feature.append(max(psdbeta))
    return feature 


def participant_affective(raw):
    a = np.zeros((23, 18, 9), dtype=object)
    for participant in range(0, 23):
        for video in range(0, 18):
            a[participant, video, 0] = (raw["DREAMER"][0, 0]["Data"]
                                        [0, participant]["Age"][0][0][0])
            a[participant, video, 1] = (raw["DREAMER"][0, 0]["Data"]
                                        [0, participant]["Gender"][0][0][0])
            a[participant, video, 2] = int(participant+1)
            a[participant, video, 3] = int(video+1)
            a[participant, video, 4] = ["Searching for Bobby Fischer",
                                        "D.O.A.", "The Hangover", "The Ring",
                                        "300", "National Lampoon\'s VanWilder",
                                        "Wall-E", "Crash", "My Girl",
                                        "The Fly", "Pride and Prejudice",
                                        "Modern Times", "Remember the Titans",
                                        "Gentlemans Agreement", "Psycho",
                                        "The Bourne Identitiy",
                                        "The Shawshank Redemption",
                                        "The Departed"][video]
            a[participant, video, 5] = ["calmness", "surprise", "amusement",
                                        "fear", "excitement", "disgust",
                                        "happiness", "anger", "sadness",
                                        "disgust", "calmness", "amusement",
                                        "happiness", "anger", "fear",
                                        "excitement", "sadness",
                                        "surprise"][video]
            a[participant, video, 6] = int(raw["DREAMER"][0, 0]["Data"]
                                           [0, participant]["ScoreValence"]
                                           [0, 0][video, 0])
            a[participant, video, 7] = int(raw["DREAMER"][0, 0]["Data"]
                                           [0, participant]["ScoreArousal"]
                                           [0, 0][video, 0])
            a[participant, video, 8] = int(raw["DREAMER"][0, 0]["Data"]
                                           [0, participant]["ScoreDominance"]
                                           [0, 0][video, 0])
    b = pd.DataFrame(a.reshape((23*18, a.shape[2])),
                     columns=["age", "gender", "participant",
                              "video", "video_name", "target_emotion",
                              "valence", "arousal", "dominance"])
    return b

def feat_extract_EEG(raw):
    EEG_tmp = np.zeros((23, 18, 42))
    for participant in range(0, 23):
        for video in range(0, 18):
            for i in range(0, 14):
                B, S = [], []
                basl = (raw["DREAMER"][0, 0]["Data"]
                        [0, participant]["EEG"][0, 0]
                        ["baseline"][0, 0][video, 0][:, i])
                stim = (raw["DREAMER"][0, 0]["Data"]
                        [0, participant]["EEG"][0, 0]
                        ["stimuli"][0, 0][video, 0][:, i])
                B = preprocess_EEG(basl, B)
                S = preprocess_EEG(stim, S)
                Extrod = np.divide(S, B)
                EEG_tmp[participant, video, 3*i] = Extrod[0]
                EEG_tmp[participant, video, 3*i+1] = Extrod[1]
                EEG_tmp[participant, video, 3*i+2] = Extrod[2]
    col = []
    for i in range(0, 14):
        col.append("psdtheta_"+str(i + 1)+"_un")
        col.append("psdalpha_"+str(i + 1)+"_un")
        col.append("psdbeta_"+str(i + 1)+"_un")
    data_EEG = pd.DataFrame(EEG_tmp.reshape((23 * 18,
                                             EEG_tmp.shape[2])), columns=col)
    scaler = StandardScaler()
    for i in range(len(col)):
        data_EEG[col[i][:-3]] = scaler.fit_transform(data_EEG[[col[i]]])
    data_EEG.drop(col, axis=1, inplace=True)
    return data_EEG


def feat_extract_ECG(raw):
    data_ECG = {}
    for participant in range(0, 23):
        for video in range(0, 18):
            # load raw baseline and stimuli data for left and right
            basl_left = (raw["DREAMER"][0, 0]["Data"]
                         [0, participant]["ECG"][0, 0]
                         ["baseline"][0, 0][video, 0][:, 0])
            stim_left = (raw["DREAMER"][0, 0]["Data"]
                         [0, participant]["ECG"][0, 0]
                         ["stimuli"][0, 0][video, 0][:, 0])
            basl_right = (raw["DREAMER"][0, 0]["Data"]
                          [0, participant]["ECG"][0, 0]
                          ["baseline"][0, 0][video, 0][:, 1])
            stim_right = (raw["DREAMER"][0, 0]["Data"]
                          [0, participant]["ECG"][0, 0]
                          ["stimuli"][0, 0][video, 0][:, 1])
            # process with neurokit
            signals_b_l, info_b_l = nk.ecg_process(basl_left,
                                                   sampling_rate=256)
            signals_s_l, info_s_l = nk.ecg_process(stim_left,
                                                   sampling_rate=256)
            signals_b_r, info_b_r = nk.ecg_process(basl_right,
                                                   sampling_rate=256)
            signals_s_r, info_s_r = nk.ecg_process(stim_right,
                                                   sampling_rate=256)
            # divide stimuli features by baseline features
            # would be interesting to compare classification accuracy when we
            # don't do this
            features_ecg_l = nk.ecg_intervalrelated(signals_s_l) / nk.ecg_intervalrelated(signals_b_l)
            features_ecg_r = nk.ecg_intervalrelated(signals_s_r) / nk.ecg_intervalrelated(signals_b_r)
            # average left and right features
            # would be interesting to compare classification accuracy when we
            # rather include both left and right features
            features_ecg = (features_ecg_l + features_ecg_r)/2
            if not len(data_ECG):
                data_ECG = features_ecg
            else:
                data_ECG = pd.concat([data_ECG, features_ecg],
                                     ignore_index=True)
    return data_ECG