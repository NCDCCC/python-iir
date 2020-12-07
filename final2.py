import wave
from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.signal
import os

#save imgs if SAVEFIG is True
SAVEFIG = False

#open original wavfile
voice = wave.open("./data/中央人民广播电台新闻报纸和摘要9s.wav", "rb")
voice_with_noise = wave.open("./data/中央人民广播电台新闻报纸和摘要电流干扰9s.wav", "rb")

outputParameters = []

def lowpass(fft, nframes):
	'''
	Deal the signal from freq domain by simply initial to 0
	Input:
		fft: output of FFT
		nframes: sampling number of wave
	Output:
		A filtered freq-domain data(array with nframes length) 

	'''
	f = np.linspace(0, 48000, nframes)
	return np.where(f < 3500, wave, 0)

def iirLowPass(wave, fp, fs, gp, gs, framerate):
	'''
	Using iir filter to deal with original signal
	Input:
		wave: original signal (time domain)
		fp: pass freq
		fs: stop freq
		gp: The maximum loss in the passband (dB).
		gs: The minimum attenuation in the stopband (dB).
		framerate: sampling freq
	Output:
		A filtered time-domain wave(array with nframes length) 
	'''
	N, Wn = scipy.signal.buttord(
				fp*2/framerate,
				fs*2/framerate,
				gp, 
				gs
			)
	print(N)
	print(Wn)
	b, a = scipy.signal.butter(N, Wn, 'lowpass')
	print(b)
	print(a)
	low_pass_data = scipy.signal.lfilter(
				b, a, wave
			)
	#low_pass_data = scipy.signal.filtfilt(
	#		b, a, wave
	#	)
	return low_pass_data

#save imgs
targetPath = './'
if SAVEFIG is True:
	if not os.path.exists(targetPath+'outputImg2'):
		os.mkdir(targetPath+'outputImg2')
	targetPath += 'outputImg2/'


titles_time_domain = ['voice_time_domain', 'voice_with_noise_time_domain']
titles_freq_domain = ['voice_freq_domain', 'voice_with_noise_freq_domain']
for v in [voice, voice_with_noise]:
	#get original wave data
	params = v.getparams()
	channels, sampwidth, framerate, nframes = params[:4]
	print(channels)
	print(sampwidth)
	print(framerate)
	print(nframes)

	#get time domain data
	strData = v.readframes(nframes)
	waveData = np.fromstring(strData, dtype=np.short)
	#print(waveData.shape)
	waveData.shape = -1, 2
	#print(waveData.shape)
	waveData = waveData.T[0]#left track
	waveData = waveData*1.0/(np.max(np.abs(waveData)))

	time = np.arange(0, nframes)*(1.0 / framerate)

	#plot time domain
	plt.figure()
	plt.plot(time, waveData)
	plt.xlabel("Time/s")
	plt.ylabel("NormedAmplitude")
	plt.title(titles_time_domain[[voice, voice_with_noise].index(v)])
	plt.grid()
	if SAVEFIG is True:
		plt.savefig(targetPath+titles_time_domain[[voice, voice_with_noise].index(v)])

	#fft to frequency domain
	nfft = nframes
	waveFFT = scipy.fft.fft(waveData, n=nfft)
	waveFFT_shift = scipy.fft.fftshift(waveFFT)	
	waveFFT_norm = np.abs(waveFFT_shift) / np.max(np.abs(waveFFT_shift))
	waveFFT_half = waveFFT_norm[int(nfft/2):]
	
	#plot freq domain
	freq = np.linspace(0, framerate/2, int(nframes/2)+1)
	plt.figure()
	plt.plot(freq, waveFFT_half)
	plt.xlabel("Freq/Hz")
	plt.ylabel("NormedAmplitude")
	plt.title(titles_freq_domain[[voice, voice_with_noise].index(v)])
	plt.grid()
	if SAVEFIG is True:
		plt.savefig(targetPath+titles_freq_domain[[voice, voice_with_noise].index(v)])

	#iir and plot
	#waveTime = scipy.fft.ifft(waveFFT, n=nframes)
	'''
	waveIir = lowpass(waveFFT, nframes)
	print('iir'+str(waveIir.shape))
	waveIir_shift = scipy.fft.fftshift(waveIir)
	waveIir_norm = np.abs(waveIir_shift)
	waveIir_half = waveIir_norm[int(nfft/2):]
	plt.figure()
	plt.plot(freq, waveIir_half)
	plt.xlabel("Freq/Hz")
	plt.ylabel("NormedAmplitude")
	plt.title('voice_with_noise_iir_freqdomain')
	plt.grid()
	'''
	#waveIir = lowpass(waveFFT, nframes)
	waveIir = iirLowPass(waveData, 3000, 4000, 1, 60, framerate)
	print('iir'+str(waveIir.shape))
	waveIir = waveIir*1.0/(np.max(np.abs(waveIir)))
	plt.figure()
	plt.plot(time, waveIir)
	plt.xlabel("Time/s")
	plt.ylabel("NormedAmplitude")
	plt.title(titles_time_domain[[voice, voice_with_noise].index(v)]+'_iir')
	plt.grid()
	if SAVEFIG is True:
		plt.savefig(targetPath+'voice_with_noise_iir_timedomain')

	#ifft and plot
	#waveIir_ishift = scipy.fft.ifftshift(waveIir)
	
	waveIirTime = scipy.fft.fft(waveIir)
	print('iirtime'+str(waveIirTime.shape))
	waveIirTime_shift = scipy.fft.fftshift(waveIirTime)
	waveIirTime_norm = np.abs(waveIirTime_shift) / np.max(np.abs(waveIirTime_shift))
	waveIirTime_half = waveIirTime_norm[int(nfft/2):]

	plt.figure()
	#time2 = np.arange(0, nframes)*(1.0 / framerate)

	plt.plot(freq, waveIirTime_half)
	plt.xlabel("Freq/Hz")
	plt.ylabel("NormedAmplitude")
	plt.title(titles_freq_domain[[voice, voice_with_noise].index(v)]+'_iir')
	plt.grid()
	if SAVEFIG is True:
		plt.savefig(targetPath+'voice_with_noise_iir_freqdomain')


voice.close()
voice_with_noise.close()

if __name__ == '__main__':
	plt.show()
	#generate new wavfile
	'''
	outWave = waveIirTime.T
	outWave.reshape(-1)
	outputParameters = params[:4]
	print(outputParameters)
	outputFilename = "./data/output.wav"
	outputFile = wave.open(outputFilename, 'wb')
	outputFile.setnchannels(2)
	outputFile.setsampwidth(outputParameters[1])
	outputFile.setframerate(outputParameters[2])
	outputFile.setnframes(outputParameters[3])
	outputFile.writeframes(outWave)
	outputFile.close()
	'''	
