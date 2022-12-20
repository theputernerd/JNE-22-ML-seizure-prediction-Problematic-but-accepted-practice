import matplotlib.pyplot as plt
import numpy as np
import scipy.fftpack
import collections

from helpers import *

from scipy.interpolate import interp1d

if __name__ == '__main__':
	####Demonstration of slow moving noise arising from mains power

	central_freq=400 #in hertz
	chan=22
	plus_minus=2 #in hertz
	minfrange = central_freq-plus_minus
	maxfrange = central_freq+plus_minus
	time_window_s= 1*60  ##5 minute time window
	mv_av_mins=60
	exp_name = f"a0f66459"
	#exp_name="chb01"
	#base_folder = f"D:\\scratch\\physionet.org\\files\chbmit\\1.0.0\\{exp_name}"
	base_folder = f"D:\scratch\\{exp_name}"

	path = f"results/mains_noise"
	
	os.makedirs(f"{path}",exist_ok=True)
	
	m = MNEDatasetManager(step_time_sec=time_window_s, window_time_sec=time_window_s, samplerate=None, basefolder=base_folder,resample=None)
	sfreq=m.samplerate
	
	T = 1.0 / sfreq
	print(f"sfreq:{sfreq}Hz")
	
	last=False
	ts=[]
	xs=[]
	addedtime=0 #everyfile
	lastt=0
	mag_maxFs=[]
	peakFreqs=[]
	counter=0
	time_h=[]
	maxlen= int(np.ceil(mv_av_mins * 60 / time_window_s)) + 1
	mags_mv_av=collections.deque(maxlen=maxlen)   #want a 10 minute moving average 10*batch_window_s/60
	mags_mv_avpts=[]
	freq_mv_av = collections.deque(maxlen=maxlen)  # want a 10 minute moving average 10*batch_window_s/60
	freq_mv_avpts = []

	#fig, axes = plt.subplots(1,sharex=True)
	count=0
	fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True)
	doFFT=True
	while True:
		try:
			data, times, done, file = m.get_next_window()
			#d=data[chan]
			if counter==0:
				T = 1.0 / sfreq
				N = len(times)
				freqs = scipy.fftpack.fftfreq(N, T)[:N // 2]
			y = np.absolute(scipy.fftpack.fft(data[chan])[:N // 2])
			
			if doFFT and len(time_h)>0 and time_h[-1]>0.25: #15 mins in create the FFT
				doFFT=False
				fig2, axes2 = plt.subplots(ncols=1, nrows=1)
				print(f"creating FFT")
	
				axes2.stem(freqs,y,markerfmt=" ", basefmt=" ",use_line_collection=True)
				plt.title(f"FFT of 1 time window for channel {chan} {exp_name}")
				#plt.xlim(xmin=0,xmax=int(256/2))
				plt.ylim(ymax=.1,ymin=0)
				plt.xlabel(f"Freq (Hz)")
				plt.ylabel(f"Amplitude")
				plt.grid()
				plt.savefig(f"{path}/FFT{exp_name}_chan{chan}_DIFF.jpg")
				plt.savefig(f"{path}/FFT{exp_name}_chan{chan}_DIFF.pdf")
				print(f"Plots saved")
				fig2.clf()
				plt.close()
			
			####get mag around power
			powidx = (np.abs(freqs - central_freq)).argmin()
			minidx= (np.abs(freqs - (central_freq- plus_minus))).argmin()
			maxidx= (np.abs(freqs - (central_freq+ plus_minus))).argmin()

			#minidx=powidx-plus_minus
			#maxidx=powidx+plus_minus# find the index of 60Hz

			mains_range= y[minidx:maxidx]
			#get max freq about 60Hz
			peakFreq_idx=np.argmax(mains_range)
			
			mag_maxF=np.absolute(mains_range[peakFreq_idx])
			
			mag_maxFs.append(mag_maxF)
			mags_mv_av.append(mag_maxF)
			if len(mags_mv_av)<maxlen:
				mags_mv_avpts.append(None)
			else:
				mags_mv_avpts.append(np.mean(mags_mv_av))

			peakFreq=freqs[minidx + peakFreq_idx]
			freq_mv_av.append(peakFreq)
			if len(freq_mv_av)<maxlen or None in freq_mv_av:
				freq_mv_avpts.append(None)
			else:
				freq_mv_avpts.append(np.mean(freq_mv_av))

			peakFreqs.append(peakFreq)
			time_h.append(counter * time_window_s / 60 / 60)

			plt.sca(axes[0])
			plt.cla()
			plt.sca(axes[1])
			plt.cla()
			
			#plt.grid()
			axes[0].set_title(f"Maximum Frequency Component between {minfrange}-{maxfrange}Hz \n {exp_name} Channel {chan} ")
			axes[0].plot(time_h, mag_maxFs, color="lightsteelblue")
			###dont plot where mags_mv_avpts has None - because it didn't have enough to calculate the full average
			mvav1=axes[0].plot(time_h,mags_mv_avpts,color="darkred")
			axes[0].set_ylabel(f"Maximum \n Amplitude")
			axes[0].ticklabel_format(axis='y',style='sci', scilimits=(0,0))
			axes[0].grid()
			try:
				axes[0].set_ylim(ymax=max(filter(None.__ne__, mags_mv_avpts)),ymin=min(filter(None.__ne__, mags_mv_avpts)))
			except:
				pass
				
			#axes[0].set_ylim(ymax=1e3,ymin=0)
			#axes[0].set_ylim(ymax=5e2, ymin=0)
			
			#ax2 = ax1.twinx()
			axes[1].plot(time_h,peakFreqs,color="lightsteelblue")
			axes[1].plot(time_h,freq_mv_avpts,color="darkred")
			axes[1].set_ylabel(f"Frequency of \n Max Amplitude (Hz)")
			#axes[1].set_ylim(ymin=59.9,ymax=60.1)
			#axes[1].set_ylim(ymin=19.985,ymax=20.015)
			try:
				axes[1].set_ylim(ymax=max(filter(None.__ne__, freq_mv_avpts)),ymin=min(filter(None.__ne__, freq_mv_avpts)))
			except:
				pass
			axes[1].ticklabel_format(axis='y',style='plain', scilimits=(-5,6))

			axes[1].grid()

			fig.legend(mvav1, [f"{mv_av_mins} min Moving Average"], loc='center')
			
			plt.xlabel("Time (h)")
			#fig.tight_layout()
			#plt.show()
			plt.pause(0.01)
			
			counter+=1
			if time_h[-1]>24:  #just plot 24H
				break

		except:
			#plt.savefig(f"ERRORplot{chan}.jpg")
			raise
			break


	plt.savefig(f"{path}/{exp_name}_{minfrange}-{maxfrange}Hz_chan{chan}.pdf")
	plt.savefig(f"{path}/{exp_name}_{minfrange}-{maxfrange}Hz_chan{chan}.jpg")

	#plt.show()
	plt.close()
	
	fig, axes = plt.subplots(ncols=1, nrows=2, sharex=True)
	
	
