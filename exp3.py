import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.layers import Flatten
from sklearn.metrics import roc_curve,auc

import mne
from scipy.fft import rfft, rfftfreq
import glob
import os

#
tf.keras.backend.clear_session()

class MNEDatasetManager:
	def __init__(self, step_time_sec=.5, window_time_sec=1, samplerate=None,
				 basefolder="EDFs"):
		"""
		:param step_time_sec: How big a step do we take every datapoint in seconds
		:param window_time_sec: How big is the window for the datapoint in seconds
		:param samplerate:
		:param lastSampleAction: 0: loadNextEdf. 1. looparound.
		"""
		self.basefolder = basefolder
		self.initialise(step_size_t=step_time_sec, window_size_t=window_time_sec, samplerate=samplerate)

	def initialise(self, step_size_t=.5, window_size_t=1, edf_idx=None, samplerate=None):
		print(f"{self.basefolder}")
		self.edffiles = glob.glob(os.path.join(self.basefolder, "*.edf")) #get list of all edfs
		if edf_idx == None:
			self.edf_idx = -1
		else:
			self.edf_idx = edf_idx - 1  ##-1 because next_edf adds 1
		if samplerate != None: self.samplerate = samplerate
		self.next_edf()
		self.step_size_t = step_size_t  # this how much we increment step in seconds.
		self.window_size_t = window_size_t  # this is how wide the window is in seconds.
		self.step_size_s = int(step_size_t*self.samplerate)  # this how much we increment step in samples.
		self.window_size_s = int(window_size_t *self.samplerate)  # this is how wide the window is in samples.

		self.idx = 0  # this is how many time steps have been completed.

	def init_edf(self, samplerate=None):
		# these are the things that might be different with each EDF.
		# call everytime the edf changes
		self.last_sample = self.edfdata.last_samp
		self.info = self.edfdata.info  # https://mne.tools/stable/generated/mne.io.read_raw_edf.html
		self.channels = self.edfdata.ch_names
		self.rawtimes = self.edfdata.times  ##all the timestepsp in this file
		#raw_data, times = self.edfdata.get_data(start=0, stop=10, return_times=True)  # get 10 samples
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

		#self.layout = mne.channels.read_layout('biosemi')

	def next_edf(self, set_idx=0):
		self.edf_idx += 1
		if self.edf_idx >= len(self.edffiles):
			print(f"FINISHED  last EDF ")
			return None,self.edf_idx
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

		data, times,last = self.get_data_samples(start_sample, start_sample + duration)


		if last: #we have finished this edf so load the next one
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
			last=False
		diff=finish-start
		raw_data, times = self.edfdata.get_data(start=start, stop=finish, return_times=return_times)
		if len(times) < diff:
			#print(f"We have grabbed the tail end of the data - it is not a full size")
			last = True

		return raw_data, times,last
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
	def get_fft(self,data,nbins=None):
		if nbins==None:
			fft_vals = rfft(data)
			fft_freqs = rfftfreq(len(data[0]), 1/self.samplerate) #freqs are tge same for each channel
		else:
			fft_vals = rfft(data,n=nbins*2)
			fft_freqs = rfftfreq(nbins*2, 1/self.samplerate) #freqs are tge same for each channel
		return fft_vals,fft_freqs

from helpers import *

def model(channels,fft_size=1281):
	model = Sequential()
	
	model.add(Dense(10,input_shape=(channels, fft_size, 1)))  # model.add(layers.Dense(noisedim, use_bias=False))
	model.add(LeakyReLU())
	model.add(Flatten())
	
	model.add(Dense(2,))  # model.add(layers.Dense(noisedim, use_bias=False))
	model.add(LeakyReLU())
	#model.add(Dropout(0.5))
	model.add(Dense(1,activation='sigmoid'))

	opt = Adam(learning_rate=0.0005, )
	model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model

class TrainingPlot(keras.callbacks.Callback):
	
	def __init__(self, folder,exp_name,plottitle=""):
		self.folder = folder
		self.exp_name=exp_name
		self.plottitle=plottitle
	def get_marker(self, change):
		if np.isnan(change):
			return ""
		if change > -1 and change < 1:
			marker = "\u25b7"
		elif change >= 1 and change < 5:  ##inprovement between 1 and 5%
			marker = "\u25b5"
		elif change >= 5 and change < 10:  ##inprovement between 1 and 5%
			marker = "\u25b4"
		elif change >= 10 and change < 30:  ##inprovement between 1 and 5%
			marker = "\u25b3"
		elif change >= 30:  ##inprovement between 1 and 5%
			marker = "\u25b2"
		elif change <= -1 and change > -5:  ##inprovement between 1 and 5%
			marker = "\u25bf"
		elif change <= -5 and change > -10:  ##inprovement between 1 and 5%
			marker = "\u25be"
		elif change <= -10 and change > -30:  ##inprovement between 1 and 5%
			marker = "\u25bd"
		elif change <= -30:  ##inprovement between 1 and 5%
			marker = "\u25bc"
		
		return marker
	
	def moving_average_comparison(self, n=20):
		"""
		compares the moving average of the last n with the preceding n
		"""
		size = len(self.losses)
		if size < 2:
			if size > 0:
				print(f"\n@report: {self.epoch}. trg:{round(self.losses[-1], 6)}\t "
				      f"test:{round(self.val_losses[-1], 6)} \t "
				      f" ", flush=True)
			return
		if size < 2 * n:
			##then it's not long enough to do the comparison. Instead just split it in two.
			latest = [int(size / 2), size]
			earliest = [0, int(size / 2)]
		else:
			latest = [size - n, size]
			earliest = [size - 2 * n, size - n]
		if size == 2:
			# print(f"Size ==2")
			pass
		av1 = np.average(self.losses[earliest[0]:earliest[1]])
		av2 = np.average(self.losses[latest[0]:latest[1]])
		change = (av2 - av1) / (av1) * 100
		marker = self.get_marker(change)
		vav1 = np.average(self.val_losses[earliest[0]:earliest[1]])
		vav2 = np.average(self.val_losses[latest[0]:latest[1]])
		valchange = (vav2 - vav1) / (vav1) * 100
		valmarker = self.get_marker(valchange)
		##TODO: Print the last 10 change markers
		print(f"\n@report: {self.epoch}. trg:{round(self.losses[-1], 6)}{marker} \t "
		      f"test:{round(self.val_losses[-1], 6)}{valmarker} \t "
		      f"change: {round(change, 1)},{round(valchange, 1)} ", flush=True)
	
	# This function is called when the training begins
	def on_train_begin(self, logs={}):
		# Initialize the lists for holding the logs, losses and accuracies
		self.losses = []
		self.acc = []
		self.val_losses = []
		self.val_acc = []
		self.logs = []
		self.epoch = 0
	
	# This function is called at the end of each epoch
	def on_epoch_end(self, epoch, logs={}):
		self.epoch = epoch
		self.moving_average_comparison()
		# Append the logs, losses and accuracies to the lists
		self.logs.append(logs)
		self.losses.append(logs.get('loss'))
		self.acc.append(logs.get('accuracy'))
		self.val_losses.append(logs.get('val_loss'))
		self.val_acc.append(logs.get('val_accuracy'))

		# Before plotting ensure at least 2 epochs have passed
		if len(self.losses) > 1:
			N = np.arange(0, len(self.losses))
			# Plot train loss, train acc, val loss and val acc against epochs passed
			plt.plot(N, self.losses, label="train_loss")
			plt.plot(N, self.val_losses, label="val_loss")
			plt.title(f"{self.exp_name}_Losses")
			plt.xlabel("Epoch")
			plt.ylabel("Loss")
			plt.legend()
			plt.grid(color='gray')

			try:
				plt.savefig(f'{self.folder}/{self.exp_name}_losses.png')
				plt.close()
			
			except:
				pass
				plt.close()
			
			#########plot accuracy
			plt.plot(N, self.acc, label="Training Accuracy")
			plt.plot(N, self.val_acc, label="Validation Accuracy")
			plt.title(f"Accuracy {self.exp_name}")
			plt.xlabel("Epoch")
			plt.ylabel("Accuracy")
			plt.legend()
			plt.grid(color='gray')
			
			try:
				plt.savefig(f'{self.folder}/{self.exp_name}_accuracy.png')
				plt.close()
			
			except:
				pass
				plt.close()
	
	def on_train_batch_end(self, batch, logs=None):
		pass

def get_X_data(m,_data,zero_dc=True):
	_data = np.array(_data)[1:24, :]  # if you want to focus on a channel do it here.
	_fft_val, _freqs = m.get_fft(_data)
	_eeg_data = _fft_val
	_eeg_mag = np.abs(_eeg_data)
	eeg_angle = np.angle(_eeg_data)
	DC=np.where(_freqs<1)
	if zero_dc:
		_eeg_mag[:, DC] = 0
		
	for ch in range(_eeg_mag.shape[0]):
		_min = _eeg_mag[ch, :].min()
		_max = _eeg_mag[ch, :].max()
		if (_max - _min) != 0:
			_normalised = (2 * (_eeg_mag[ch, :] - _min) / (_max - _min)) - 1
			_eeg_mag[ch, :] = _normalised

	_eegX = _eeg_mag
	return _eegX, _freqs

def get_labels(window_sec, step_sec, folder, switch_every_min=60, skip=2,internal_skip=False, max_number_of_switches=100):
	###skip is how many hours to skip between labelled period
	m = MNEDatasetManager(step_time_sec=step_sec, window_time_sec=window_sec, basefolder=folder)  # , basefolder='new_test_set'
	Xs = []
	Ys = []
	Ts = []
	info = m.info
	channel_count = info['nchan']
	samplerate = m.samplerate
	expected_samples = int(window_sec * samplerate)

	edfFile,_=m.current_edf()
	edfName=os.path.basename(edfFile)
	next_switch=0#convert the switch time to seconds

	category=100 #start with a big number so it is reset the first time through the loop
	
	labels_counter=0
	window_counter=0
	#	=0
	while True:
		try:
			data, times, done, file = m.get_next_window()
			#last_time_in_edf=m.rawtimes[-1]
			cum_time=window_counter*m.step_size_s/m.samplerate +m.window_size_t  #could use m.step_size_sec
			window_counter+=1
			if labels_counter>max_number_of_switches:
				print(f"finished. Have labelled {labels_counter}X2 different windows")
				break
			
			if file==None:
				break ###we've finished all files.
			if cum_time>=next_switch:
				next_switch += switch_every_min*60
				print(f"{round(cum_time/60/60,2)}H")
				category=category+1
				if category>skip+1:
					labels_counter+=1
					category=0
					
			if category<=1 and (len(times) == expected_samples):  #only save data if category is a known category
				eegX, freqs = get_X_data(m,data)
				Xs.append(eegX)
				Ys.append(category)
				#Ys.append(random.choice([0,1]))
				
				#Ts.append(data) #if you want to save memory don't save the time.

		except:
			raise
			
	return Xs, Ys, Ts, freqs

import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':

	window_sec = 10  # 1.875
	step_sec = 10
	singleChan = False
	patient = "chb02"
	edfs=f"D:\\scratch\\physionet.org\\files\\chbmit\\1.0.0\\{patient}"
	
	#patient = "a0f66459"
	#edfs = f"D:\scratch\\{patient}"
	folder = os.path.abspath(edfs)
	path=f"results\\ml\\{patient}"
	switch_every_mins=[240,10,120,60]
	#switch_every_mins=[240,60,10]
	step_secs=[10]
	skips=[5,2,0]
	#skips=[5,0]
	max_number_of_window_labels=100 #i.e. how many "seizures"
	val_acc=[]  ##skip,val
	test_acc = []  ##skip,test
	rocstats=[]
	Ds=[]
	As=[]
	for step_sec in step_secs:
		for switch_every_min in switch_every_mins:
			for skip in skips:
				exp_name=f"P={switch_every_min}min N={skip}"
				Xs, Ys, Ts, freqs = get_labels(window_sec, step_sec, folder,
				                               max_number_of_switches=max_number_of_window_labels, switch_every_min=switch_every_min, skip=skip)
				powidxs = (np.abs(freqs - 60)).argmin()  # used to select only freqs around the power

				proportion=np.sum(Ys)/len(Ys)
				print(f"Label Proportion: {proportion}")

				Xs, Ys, Ts = np.array(Xs), np.array(Ys), np.array(Ts)
				Xs = np.expand_dims(Xs, axis=-1)
				
				X_train, X_test, Y_train, Y_test = train_test_split(Xs, Ys, test_size=0.05)
				X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)
				
				fX=X_train[0][0]
				os.makedirs(path, exist_ok=True)
				plt.plot(freqs,np.squeeze(fX))
				plt.xlim(0.5)
				plt.savefig(f"{path}/FFT_{exp_name}.png")
				plt.close()
				CNN_model = model(channels=X_train.shape[1],fft_size=len(freqs))

				ES = tf.keras.callbacks.EarlyStopping(
					monitor="val_loss",
					min_delta=0,
					patience=20,
					verbose=0,
					mode="auto",
					baseline=None,
					restore_best_weights=True,
				)
				lossplots = TrainingPlot(path,exp_name)

				try:
					history = CNN_model.fit(X_train, Y_train, batch_size=5, validation_data=(X_val, Y_val), epochs=50,
					                        verbose=True,
					                        use_multiprocessing=True, callbacks=[ES, lossplots])
				except KeyboardInterrupt:
					
					pass
				y_pred_keras = CNN_model.predict(X_test).ravel()
				
				fpr_keras, tpr_keras, thresholds_keras = roc_curve(Y_test, y_pred_keras)
				auc_keras = auc(fpr_keras, tpr_keras)
				print(auc_keras)
				rocplotname=f"{path}/AUC"
				D = (2 * switch_every_min + skip * switch_every_min) / 2
				Ds.append(D/60)
				As.append(auc_keras)
				rocstats.append([D,fpr_keras, tpr_keras, auc_keras,f"D={round(D/60,2)} H "])
				plot_rocs(rocstats=rocstats,filename=rocplotname)
				
				val_acc.append([skip,max(history.history['val_accuracy'])])
				test_acc.append([skip,max(history.history['accuracy'])])
				
				row=[D,skip,switch_every_min,step_sec,max(history.history['val_accuracy']),max(history.history['accuracy'])]
				Ds, As = zip(*sorted(zip(Ds, As)))
				Ds, As = list(Ds),list(As)
				plt.plot(Ds,As)
				plt.grid()
				plt.title(f"Performance dependence on class distance ")
				plt.ylabel(r"Area under the ROC curve")
				plt.xlabel(r"Approx. Distance between classes $D=\frac{(2P+NP)}{2}$ (Hours)")
				plt.savefig(f"{path}/DvsA.png")
				plt.savefig(f"{path}/DvsA.pdf")
				plt.close()
				with open(f'{path}/performance.csv', 'a', newline="") as f:
					write = csv.writer(f)
					write.writerow(row)
			
		




