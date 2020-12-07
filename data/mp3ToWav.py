import pydub
import wave
import os

filepath = './'
filename = os.listdir(filepath)
print(filename)

for file in filename:
	if 'py' not in file and 'mp3' in file:
		print(file)
		voice_mp3 = pydub.AudioSegment.from_mp3(filepath+file)
		name = file.split('.')[0]
		voice_mp3.export(filepath+name+".wav", format="wav")