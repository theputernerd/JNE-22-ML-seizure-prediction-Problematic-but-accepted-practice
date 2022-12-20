import collections
import datetime

from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from collections import deque
import scipy
from helpers import *
from itertools import combinations

#define length of step/batch in seconds
window_sec = 10 # 1.875
step_sec = 10

import numpy as np

def plotstuff(t,metric,mvav_n=700,label="Euc",plotname="Euclidean Distance",do_pdf=False):
    plt.close()
    t_sorted, met_sorted = zip(*sorted(zip(t, metric)))
    average_chan_dist=[]
    av_t=[]
    mvaves=deque(maxlen=mvav_n)
    plotted_mvaves=[]
    for ind in range(len(t)):
        av=np.mean(met_sorted[ind])
        average_chan_dist.append(av)
        mvaves.append(np.mean(av))
        if len(mvaves)==mvav_n:
            plotted_mvaves.append(np.mean(mvaves))
        else:
            plotted_mvaves.append(np.nan)

        av_t.append(np.mean(t_sorted[ind]))
    #plt.plot(np.array(t_sorted) / 60/60, met_sorted, alpha=0.1)
    plt.plot(np.array(av_t) / 60 / 60, average_chan_dist,alpha=0.2)
    legend2=plt.plot(np.array(av_t) / 60 / 60, plotted_mvaves,color="darkred")

    plt.title(plotname)
    plt.xlabel("Time Difference (H)")
    plt.ylabel("Distance")
    plt.xticks(range(0, 1+int(np.ceil(av_t[-1]/60/60))))
    #plt.yticks(np.arange(0, 2,0.01))

    plt.xlim(left=0)
    plt.grid(which='both')
    plt.legend(legend2,[f"{mvav_n} Point Moving Average"])
    try:
        ymin = np.nanmin(plotted_mvaves)
        ymax = np.nanmax(plotted_mvaves)
        print(f"{ymax},{ymin}")
        plt.ylim(ymin,ymax)
    except:
        pass
    
    #plt.ylim(top=0.1,bottom=0)
    os.makedirs("results\\dist",exist_ok=True)
    plt.savefig(f"results\\dist\\{label}.jpg")
    if do_pdf:
        plt.savefig(f"results\\dist\\{label}.pdf")
    
    #plt.ylim([.96,.98])
    #plt.show()

def similarityFn(X_old, X_new):
    euc,cos_sim,corr=[],[],[]
    for a, b in zip(X_old, X_new):
        euc.append(scipy.spatial.distance.euclidean(a, b))
        cos_sim.append(scipy.spatial.distance.cosine(a, b))
        corr.append(scipy.spatial.distance.correlation(a, b))
        
    return euc,cos_sim,corr

def get_X_data(_data,_m):
    _fft_val, _freqs = _m.get_fft(_data)
    _eeg_data = _fft_val
    _eeg_mag = np.abs(_eeg_data)
    eeg_angle = np.angle(_eeg_data)
    for ch in range(_eeg_mag.shape[0]):
        _min = _eeg_mag[ch, :].min()
        _max = _eeg_mag[ch, :].max()
        if (_max - _min) != 0:
            _normalised = (2 * (_eeg_mag[ch, :] - _min) / (_max - _min)) - 1
            _eeg_mag[ch, :] = _normalised

    _eegX = _eeg_mag
    return _eegX

if __name__ == '__main__':
    time_window_min=5   ##10 seconds
    time_window_s = time_window_min * 60  ##5 minute time window
    #chan = 5
    n=range(1,2) #for the UW SET dont iterate
    channels = list(range(14, 15))
    #channels = list(range(0, 24))
    #n=range(1,2)
    for patient in n:
        #channels = [np.random.randint(2, 14)]
        #patient=f"chb{str(pid).zfill(2)}"
        #base_folder = f"D:\\scratch\\physionet.org\\files\chbmit\\1.0.0\\{patient}"
        patient="a0f66459"
        base_folder = f"D:\scratch\\{patient}"
        #base_folder = f"D:\\scratch\\physionet.org\\files\chbmit\\1.0.0\\chb{str(patient).zfill(2)}"

        #exp_name=f"Chan{channels[0]}_{round(time_window_min,2)}_{patient}"
        #exp_name = f"chb{str(patient).zfill(2)}_{round(time_window_min, 2)}_nchans{len(channels)}"
        exp_name = f"chb{str(patient)}_{round(time_window_min, 2)}_nchans{len(channels)}"

        #path = f"results/temporal_diff"
        #os.makedirs(f"{path}",exist_ok=True)
        m = MNEDatasetManager(step_time_sec=time_window_s, window_time_sec=time_window_s, samplerate=None, basefolder=base_folder,resample=None)
        sfreq=m.samplerate
        T = 1.0 / sfreq
        delta_t_s = []
        mean_distances = []
        edf_offset=0
        ffts=[]
        ts=[]
        
        while True:
            X_current_edf = []
            t_current_edf = []
            edf_len_s = m.edfdata.times[-1]
            edf_len_m = edf_len_s / 60
            edf_len_h = edf_len_m / 60
            
            ##
            time_indexs = np.random.choice(range(0,len(m.edfdata.times)), 5000, replace=False) #ieeg cause there is few files.
            #time_indexs = np.random.choice(range(0,len(m.edfdata.times)), 200, replace=False)  ###CHB

            print(f"Made the choice now loading")
            for idx in time_indexs:
                sample_len = int(m.samplerate * window_sec)
                raw_data, times = m.edfdata.get_data(start=int(idx), stop=int(idx+sample_len), return_times=True)
                
                if len(times) != sample_len:  ##got unluck with the selection of the start
                    print(f'wrong n samples - probs because selected time + window is too long for edf')
                    continue
                raw_data=raw_data[channels]
                X = get_X_data(raw_data,m)
                ffts.append(X)
                ts.append(times[0]+edf_offset)
            ##get the next edf
            edf_offset+=edf_len_s
            file,idx=m.next_edf()
            
            if idx>len(m.edffiles) or file==None:
                print("Done alll edfs")
                break
        ###now need to find the similarity
        
        all_idx_pairs = list(combinations(range(len(ts)), 2))
        eucs,cos_sims,corrs=[],[],[]
        delta_ts=[]
        counter=0
        np.random.shuffle(all_idx_pairs)
        for idx1,idx2 in all_idx_pairs:
            counter+=1
            if counter>100000:
                break
            delta_t = abs(ts[idx1] - ts[idx2])
            if delta_t/60/60 > 24:
                continue
            delta_ts.append(delta_t)
            euc,cos_sim,corr=similarityFn(ffts[idx1],ffts[idx2])
            eucs.append(euc)
    
            if counter%20000==0:
                plotstuff(delta_ts, eucs, label=exp_name, do_pdf=True,plotname=f"FFT Euclidean Distance Patient {patient} ")
                print(f"{datetime.datetime.now()}   {counter}.  - PDF SAVED ")
                lastIdx=idx1
            elif counter%5000==0:
                plotstuff(delta_ts,eucs,label=exp_name,plotname=f"FFT Euclidean Distance Patient {patient} ")
                lastIdx=idx1
                print(f"{datetime.datetime.now()}   {counter}   - Plotted ")
    
    
        plotstuff(delta_ts, eucs, label=exp_name,do_pdf=True,plotname=f"FFT Euclidean Distance, Patient {patient} ")

        #cos_sims.append(cos_sim)
        #corrs.append(corr)
        
    
    
        
    
    
    