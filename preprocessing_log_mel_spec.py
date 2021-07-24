import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Aim - find the best method of creating log mel spectrograms

file = "blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050)

# Method 1: stft -> spectrogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)
spectrogram = np.abs(stft)

log_spectrogram = librosa.amplitude_to_db(spectrogram)
print(log_spectrogram.shape)

librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)

plt.xlabel("Time")
plt.ylabel("Frequency")
plt.colorbar()
plt.show()

# Method 2:
mel_spectrogram = librosa.feature.melspectrogram(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=256)
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
print(log_mel_spectrogram.shape)

# librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=hop_length)
#
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

librosa.display.specshow(log_mel_spectrogram,
                         x_axis="time",
                         y_axis="mel",
                        sr=sr)
plt.colorbar(format="%+2.f")
plt.show()