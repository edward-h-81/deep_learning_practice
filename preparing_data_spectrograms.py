import json
import os
import math
import librosa
from matplotlib import pyplot as plt
import librosa, librosa.display
import numpy as np

DATASET_PATH = "marsyas_mini"
JSON_PATH = "data_10_log_spec.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


def save_spectrograms(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=10):
    """Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels.

        :param dataset_path (str): Path to dataset
        :param json_path (str): Path to json file used to save MFCCs
        :param num_mfcc (int): Number of coefficients to extract
        :param n_fft (int): Interval we consider to apply FFT. Measured in # of samples
        :param hop_length (int): Sliding window for FFT. Measured in # of samples
        :param: num_segments (int): Number of segments we want to divide sample tracks into
        :return:
        """

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "labels": [],
        "log_spectrograms": []
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_spectrograms_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

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

                    # extract spectrograms

                    stft = librosa.core.stft(signal[start:finish], n_fft=n_fft,hop_length=hop_length)
                    spectrogram = np.abs(stft)

                    # log_spectrogram = librosa.amplitude_to_db(spectrogram)
                    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

                    # code I've added to check log spectrograms look okay before saving to json
                    librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length, y_axis='log', x_axis='time')
                    # librosa.display.specshow(log_spectrogram, sr=sample_rate, hop_length=hop_length)

                    # plt.xlabel("Time")
                    # plt.ylabel("Frequency")
                    plt.colorbar()
                    plt.show()

                    log_spectrogram = log_spectrogram.T

                    # store only spectrogram feature with expected number of vectors
                    if len(log_spectrogram) == num_spectrograms_vectors_per_segment:
                        data["log_spectrograms"].append(log_spectrogram.tolist())
                        data["labels"].append(i - 1)
                        print("{}, segment:{}".format(file_path, d + 1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    save_spectrograms(DATASET_PATH, JSON_PATH, num_segments=10)
