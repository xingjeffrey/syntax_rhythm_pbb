from scipy.io import wavfile
import numpy as np
import wave
import librosa


def read_wav(wav_loc, method="librosa", **kwargs):
    """ read wav using either librosa or scipy
    """
    if method == "librosa":
        if "sr" not in kwargs.keys():
            kwargs["sr"] = None
        data, rate = librosa.core.load(wav_loc, **kwargs)
    elif method == "scipy":
        rate, data = wavfile.read(wav_loc)
    return rate, data


def load_wav(wav_loc, catch_errors=True, method="librosa", **kwargs):
    if catch_errors:
        try:
            rate, data = read_wav(wav_loc, method=method, **kwargs)
            return rate, data
        except Exception as e:
            print(e)
            return None, None
    else:
        rate, data = read_wav(wav_loc, method=method, **kwargs)
        return rate, data


def write_wav(loc, rate, data):
    wavfile.write(loc, rate, data)


def int16_to_float32(data):
    """ Converts from uint16 wav to float32 wav
    """
    if np.max(np.abs(data)) > 32768:
        raise ValueError("Data has values above 32768")
    return (data / 32768.0).astype("float32")


def float32_to_uint8(data):
    """ convert from float32 to uint8 (256)
    """
    raise NotImplementedError


def float32_to_int16(data):
    if np.max(data) > 1:
        data = data / np.max(np.abs(data))
    return np.array(data * 32767).astype("int16")


def get_samplerate(file):
    with wave.open(file, "rb") as f:
        samplerate = f.getframerate()
    return samplerate

def taper_signal(signal, samps = 100):
    # ensure long enough for head and tail
    
    # if the length of the signal is less then taper length * 2
    if len(signal) < (samps*2):
        # decrease samp dynamically so that linear ramps can be executed on both sides
        samps = int(np.floor(len(signal-1)/2))
    
    # taper head and tail with linear ramping
    head = signal[:samps] * np.linspace(0,1, samps) 
    tail = signal[-samps:] * np.linspace(0,1, samps)[::-1]
    
    # merge
    signal = np.concatenate([head, signal[samps:-samps], tail])
    return signal
