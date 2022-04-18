'''
Description: 
Author: Yu Chao 
Date: 2022-04-16 11:34:36
LastEditTime: 2022-04-18 19:21:34
'''

import numpy as np
import matplotlib.pyplot as pl
from numpy.fft import fft,ifft,fftshift
from tqdm import  trange
from multiprocessing import Pool
from scipy.signal import stft
import time

from filter import matched_filter
from channel_est import channel_est


########## data params ##########
num_data_tones=60
data_count = 2184

def save_file(FileName,CSI):
    with open(FileName, 'wb') as fid:
        CSI.tofile(fid)
    print("[succ]  File saving complete.\n")

def CSI_plot(CSI,title):
    CSI_power = CSI*np.conj(CSI)
    # CSI_power = abs(CSI**2)
    CSI_sum = np.sum(CSI_power,axis=1)
    fs = 10e6/data_count
    nperseg = 512
    amp = 2 * np.sqrt(2)
    f, t, Zxx = stft(CSI_sum, fs,window='hann',nperseg=nperseg,noverlap=round(0.9*nperseg))
    # print(type(Zxx),Zxx.shape)
    pl.pcolormesh(t, f, np.abs(Zxx), shading= 'auto', cmap = 'jet', vmin=0,vmax=amp)
    pl.colorbar()
    pl.ylim(-500,0)
    pl.title(title)
    pl.ylabel('Frequency [Hz]')
    pl.xlabel('Time [sec]')
    pl.show()

def ReadFile(fname):
    data_complex=np.fromfile(fname,dtype=np.complex64,count=-1)
    # print('data size:{}'.format(data_complex.shape))
    data_sample = data_complex.reshape(-1,2,order = "F")
    data_sample = np.transpose(data_sample)
    data_ref = data_sample[0,:]
    data_ref = np.squeeze(data_ref)
    data_tar = data_sample[1,:]
    data_tar = np.squeeze(data_tar)
    return data_ref,data_tar

    
def CSI_est(data):
    data_size = len(data)
    Total_Num = int(np.floor(data_size / data_count)-1)
    CSI = np.zeros((Total_Num,num_data_tones),dtype=complex)
    ###############################
    for idx_num in trange(Total_Num-1):
        data_temp = data[idx_num*data_count:data_count*2+idx_num*data_count]
        CSI[idx_num,:] = channel_est(data_temp)
    return CSI
    

def main():
    #################   file info  #######################
    file = "x3_fast.dat"
    path = "/home/yuc/passive mmwave/eder_evk/RX/Dataset/data_0413/" 
    fname=path + file
    ################# Read file #######################
    data_ref,data_tar = ReadFile(fname)
    ################## CSI calculate  #################
    # CSI_ref = CSI_est(data_ref)
    # CSI_tar = CSI_est(data_tar)
    ################# Multi Process CSI ##############3
    t = Pool()
    res_ref = t.apply_async(CSI_est,(data_ref,))
    res_tar = t.apply_async(CSI_est,(data_tar,))
    t.close()
    #关闭pool ,不再往里加进程
    t.join()
    #等所有进程结束后,开始主进程
    CSI_ref = res_ref.get()
    CSI_tar = res_tar.get()
    #join 等两个进程结束后，主进程才继续,多进程要比多线程快，应该是因为计算太多
    print('CSI est ending')
    ############  STFT plot #######################
    ref_title = 'ref channel STFT'
    tar_title = 'tar channel STFT'
    CSI_plot(CSI_ref,ref_title)
    CSI_plot(CSI_tar,tar_title)
    ############## CSI File save ######################
    CSI_FileName_ref = path + "RefCSI_" +file 
    CSI_FileName_tar = path + "TarCSI_" +file 
    save_file(CSI_FileName_ref,CSI_ref)
    save_file(CSI_FileName_tar,CSI_tar)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('time using {:.2f}s'.format(end - start))




