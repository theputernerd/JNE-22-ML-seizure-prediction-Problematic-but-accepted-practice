import glob,os
from scipy import signal
from scipy.fft import rfft, rfftfreq,fft, fftfreq, fftshift
import mne
import numpy as np

class MNEDatasetManager:
	def __init__(self, step_time_sec=.5, window_time_sec=1, samplerate=None,
	             basefolder="EDFs", resample=None):
		"""
		:param step_time_sec: How big a step do we take every datapoint in seconds
		:param window_time_sec: How big is the window for the datapoint in seconds
		:param samplerate:
		:param lastSampleAction: 0: loadNextEdf. 1. looparound.
		"""
		self.resample = resample
		self.basefolder = basefolder
		self.initialise(step_size_t=step_time_sec, window_size_t=window_time_sec, samplerate=samplerate)
	
	def initialise(self, step_size_t=.5, window_size_t=1, edf_idx=None, samplerate=None):
		print(f"{self.basefolder}")
		self.edffiles = glob.glob(os.path.join(self.basefolder, "*.edf"))  # get list of all edfs
		if edf_idx == None:
			self.edf_idx = -1
		else:
			self.edf_idx = edf_idx - 1  ##-1 because next_edf adds 1
		if samplerate != None:
			self.samplerate = samplerate
			
		self.next_edf()
		self.step_size_t = step_size_t  # this how much we increment step in seconds.
		self.window_size_t = window_size_t  # this is how wide the window is in seconds.
		self.step_size_s = int(step_size_t * self.samplerate)  # this how much we increment step in samples.
		self.window_size_s = int(window_size_t * self.samplerate)  # this is how wide the window is in samples.
		
		self.idx = 0  # this is how many time steps have been completed.
	
	def init_edf(self, samplerate=None):
		# these are the things that might be different with each EDF.
		# call everytime the edf changes
		if self.resample != None:
			print(f"Resampling to {self.resample}Hz. This can take some time.")
			self.edfdata = self.edfdata.resample(self.resample)
		self.last_sample = self.edfdata.last_samp
		self.info = self.edfdata.info  # https://mne.tools/stable/generated/mne.io.read_raw_edf.html
		self.channels = self.edfdata.ch_names
		self.rawtimes = self.edfdata.times  ##all the timestepsp in this file
		# raw_data, times = self.edfdata.get_data(start=0, stop=10, return_times=True)  # get 10 samples
		est_samplerate = 1 / ((self.rawtimes[-1] - self.rawtimes[-10]) / 9)
		est_samplerate2 = 1 / (self.rawtimes[1] - self.rawtimes[0])
		if samplerate != None:
			self.samplerate = samplerate
		else:
			try:
				self.samplerate = self.info['sfreq']
			except:
				self.samplerate = est_samplerate2
		if est_samplerate2 != est_samplerate or est_samplerate != self.samplerate:
			if samplerate != None:
				self.samplerate = samplerate
			else:
				self.samplerate = est_samplerate2
			print(f"SAMPLE RATE CONFLICT IN {self.current_edf()} line 82 eeghelpers.py. {self.samplerate}Hz set.")
	
	# self.layout = mne.channels.read_layout('biosemi')
	
	def next_edf(self, set_idx=0):
		self.edf_idx += 1
		if self.edf_idx >= len(self.edffiles):
			print(f"FINISHED  last EDF ")
			return None, self.edf_idx
		self.edfdata = mne.io.read_raw_edf(self.edffiles[self.edf_idx])
		
		self.idx = set_idx
		self.init_edf()
		return self.edffiles[self.edf_idx], self.edf_idx
	
	def current_edf(self):
		return self.edffiles[self.edf_idx], self.edf_idx
	
	def get_next_window(self):
		# gets the next time step of data from the EEG.
		start_sample = self.idx * self.step_size_s
		duration = self.window_size_s
		
		data, times, last = self.get_data_samples(start_sample, start_sample + duration)
		
		if last:  # we have finished this edf so load the next one
			file, idx = self.next_edf()
			if file == None:
				print(f"Very last EDF processed")
				return None, None, True, None
		else:
			self.idx = self.idx + 1
		
		return data, times, last, self.edffiles[self.edf_idx]
	
	def get_data_samples(self, start, finish, return_times=True):
		if (finish) > self.last_sample:
			last = True
		else:
			last = False
		diff = finish - start
		raw_data, times = self.edfdata.get_data(start=start, stop=finish, return_times=return_times)
		if len(times) < diff:
			# print(f"We have grabbed the tail end of the data - it is not a full size")
			last = True
		
		return raw_data, times, last
	
	def get_data_time(self, start_time_mins, duration_mins, samplen=None):
		sample_start = int(60 * start_time_mins * self.samplerate)
		if samplen != None:
			sample_finish = sample_start + samplen
		else:
			sample_finish = int(np.ceil(60 * (start_time_mins + duration_mins) * self.samplerate))
		last = False
		if sample_finish >= self.last_sample:
			
			raw_data, times = self.edfdata.get_data(start=sample_start, stop=sample_finish, return_times=True)
			last = True
		else:
			# raw_data,times = self.edfdata.get_data_samples(start=sample_S,stop=sample_finish,return_times=True)
			raw_data, times = self.edfdata.get_data(start=sample_start, stop=sample_finish, return_times=True)
		
		return raw_data, times, last
	
	def get_fft(self, data, nbins=None):
		if nbins == None:
			fft_vals = rfft(data)
			fft_freqs = rfftfreq(len(data[0]), 1 / self.samplerate)  # freqs are tge same for each channel
		else:
			fft_vals = rfft(data, n=nbins * 2)
			fft_freqs = rfftfreq(nbins * 2, 1 / self.samplerate)  # freqs are tge same for each channel
		return fft_vals, fft_freqs

from scipy.interpolate import interp1d

def downsample(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled

import matplotlib.pyplot as plt

def plot_rocs(rocstats,filename):
	#plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	sortedlist=sorted(rocstats, key=lambda x: x[0]) #trying to get the legend in order of D
	for D,fpr, tpr,auc,label in sortedlist:
		plt.plot(fpr, tpr, label=label+' AUC = {:.2f}'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig(f"{filename}.png")
	plt.savefig(f"{filename}.pdf")
	plt.close()
	#plt.show()

def plot_roc(fpr, tpr,auc,filename="AUC"):
	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='(area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig(f"{filename}.png")
	plt.savefig(f"{filename}.pdf")
	plt.show()

	