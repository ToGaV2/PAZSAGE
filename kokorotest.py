from kokoro import KPipeline
#from IPython.display import display, Audio
import soundfile as sf
import torch
import numpy as np
import sys

pipeline = KPipeline(lang_code='a')

text = '''
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
'''

audiostep = []

generator = pipeline(text, voice='af_heart')
for i, (gs, ps, audio) in enumerate(generator):
    print(i)
    #print(i, gs, ps)
    #display(Audio(data=audio, rate=24000, autoplay=i==0))
    audiostep.append(audio)
print(f"Size of Audiostep: {sys.getsizeof(audiostep)}")
print("concatenating")
finaudio = np.concatenate(audiostep)
print(f"Size of FinAudio: {sys.getsizeof(finaudio)}")
print("writing output.wav")
sf.write('output.wav', finaudio, 24000)