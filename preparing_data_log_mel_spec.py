import json
import os
import math
import librosa
import numpy as np
from matplotlib import pyplot as plt
import librosa, librosa.display

DATASET_PATH = "marsyas_mini"
JSON_PATH = "data_10_log_mel_spec.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_log_mel_spec(dataset_path, json_path, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and log mel specs
    data = {
        "mapping": [],
        "labels": [],
        "log_mel_spec": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_log_mel_spec_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # loop through all genre sub-folder
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label (i.e., sub-folder name) in the mapping
            semantic_label = dirpath.split("/")[-1]
            data["mapping"].append(semantic_label)
            print("\nProcessing: {}".format(semantic_label))

            # process all audio files in genre sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # process all segments of audio file
                for d in range(num_segments):

                    # calculate start and finish sample for current segment
                    start = samples_per_segment * d
                    finish = start + samples_per_segment

                    # extract log mel spec
                    # MAYBE ADD FEATURE FROM preparing_data_spectrograms.py
                    mel_spec = librosa.feature.melspectrogram(signal[start:finish], sample_rate, n_mels=256, n_fft=2048,
                                                hop_length=hop_length)
                    log_mel_spec = librosa.amplitude_to_db(mel_spec)
                    # log_mel_spec = librosa.power_to_db(mel_spec)

                    print(log_mel_spec.shape)

                    # code I've added to check log mel specs look okay before saving to json
                    librosa.display.specshow(log_mel_spec, sr=sample_rate, hop_length=hop_length)
                    plt.xlabel("Time")
                    plt.ylabel("Frequency")
                    plt.colorbar()
                    plt.show()

                    log_mel_spec = log_mel_spec.T

                    # store only mfcc feature with expected number of vectors
                    if len(log_mel_spec) == num_log_mel_spec_vectors_per_segment:
                        data["log_mel_spec"].append(log_mel_spec.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save log mel specs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_log_mel_spec(DATASET_PATH, JSON_PATH, num_segments=10)
