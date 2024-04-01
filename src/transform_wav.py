import io
import wave

import soundfile as sf
import numpy as np

'''
Sometimes, wav files are not in the format that we expect. For example, the audio data may be in float32 format,
usually you would utilize a command line tool like ffmpeg to convert the audio to the format you want.
However, if you want to do this in python, you can use the following code snippet to convert the audio to linear16 format.
This code snippet uses the soundfile library to read the audio data and then converts it to linear16 format.
The code snippet also uses the wave library to write the audio data to a memory file object. 
'''


class WavAudio:
    def __init__(self, data: [str, bytes]) -> None:
        self.path: str = None  # Path to the audio file, if provided
        self.audio = None  # Audio data as a numpy array
        self.channels = None
        self.duration = None

        if type(data) == str:
            self.path = data
            self.read_audio()
        elif type(data) == bytes:
            self._audio_bin = data

        self.read_audio()

    def read_audio(self):
        self.audio, self.sr = sf.read(self.path)
        self.channels = 1 if len(self.audio.shape) == 1 else self.audio.shape[1]
        self.duration = len(self.audio) / self.sr

    def get_info(self):
        return {
            'channels': self.channels,
            'sample_rate': self.sr,
            'duration': self.duration
        }

    def normalize_to_linear16(self) -> [bytes, None]:
        with sf.SoundFile(io.BytesIO(self._audio_bin)) as b:

            audio = b.read()

            samplerate = b.samplerate
            nchannels = b.channels

            # Handle different audio data types

            if audio.dtype == np.int16:  # handle int16 dtype
                processed_audio = audio

            elif audio.dtype == np.int32:  # handle int32 dtype
                int32_data = np.clip(audio, -2 ** 15, 2 ** 15 - 1)
                processed_audio = (int32_data // 2 ** 16).astype(np.int16)

            elif audio.dtype == np.float32:  # handle float32 dtype
                float32_data = np.clip(audio, -1, 1)  # Assuming float32 audio is normalized
                processed_audio = (float32_data * np.iinfo(np.int16).max).astype(np.int16)

            elif audio.dtype == np.float64:  # handle float64dtype
                float64_data = np.clip(audio, -1, 1)  # Assuming float64 audio is normalized
                processed_audio = (float64_data * np.iinfo(np.int16).max).astype(np.int16)
            else:
                raise ValueError(f'Unsupported WAV datatype: {audio.dtype}')

            # Prepare a memory file object with
            mem_file = io.BytesIO()
            with wave.open(io.BytesIO(self._audio_bin), 'wb') as out_wav:
                out_wav.setnchannels(nchannels)
                out_wav.setsampwidth(2)
                out_wav.setframerate(samplerate)
                out_wav.setcomptype('NONE')
                out_wav.writeframes(processed_audio.tobytes())

            mem_file.seek(0)  # Rewind to the beginning of the memory file
            self._audio_bin = mem_file.getvalue()

        return self._audio_bin
