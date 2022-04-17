'''
Description: 
Author: Yu Chao 
Date: 2022-04-16 12:42:50
LastEditTime: 2022-04-17 15:13:13
'''

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

########### data params ###############3
fs = 10e6
duration = 3
CIT = 0.1
array_start_time = np.arange(0,duration-CIT,CIT)
Doppler_frequency = 400
array_Doppler_frequency = np.arange(-Doppler_frequency,Doppler_frequency+1/CIT,1/CIT)

### Read files
def Read_file(fname):
    data = np.fromfile(fname,dtype = '<f', count = -1,).reshape(2,-1,order = "F")
    #数据转化为两行
    # print(data.shape)
    data_complex = data[0,:] + 1j*data[1,:]
    data_sample = data_complex.reshape(-1,2,order = "F")
    # print(data_sample.shape)
    data_sample = np.transpose(data_sample)
    #第一行是ref，第二行是tar
    # num_sample = round(CIT * fs)
    data_ref = data_sample[1,:]
    data_tar = data_sample[0,:]
    # print(type(data_ref))
    plt.figure(1)
    plt.subplot(211)
    plt.plot(data_ref[:2184*2].imag)
    plt.subplot(212)
    plt.plot(data_tar[:2184*2].real)
    N  =int(2184*2)
    h_corr = np.correlate(data_tar[:N *2+1],data_ref[:N *2+1],'same')
    h_corr = abs(h_corr)
    h_corr_max = np.argmax(h_corr) 
    array_sample_shift = h_corr_max - N 
    if array_sample_shift > 0:
        data_ref = data_ref[:-array_sample_shift]
        data_tar = data_tar[array_sample_shift:]
    else:
        data_ref = data_ref[-array_sample_shift:]
        data_tar = data_tar[:array_sample_shift]
    # print(data_tar[:10],data_ref[:10])
    h_corr = np.correlate(data_tar[:N *2+1],data_ref[:N *2+1],'same')
    h_corr = abs(h_corr)
    h_corr_max = np.argmax(h_corr) 
    # print(data_ref.size,data_tar.size,array_sample_shift)
    show_max = '['+str(h_corr_max)+' '+str(h_corr[h_corr_max])+']'
    # print(h_corr_max)
    plt.figure(2)
    plt.plot(h_corr)
    plt.plot(h_corr_max,h_corr[h_corr_max],'ks')
    plt.annotate(show_max,xytext=(h_corr_max,h_corr[h_corr_max]),xy=(h_corr_max,h_corr[h_corr_max]))
    plt.title('correlate')
    # plt.show()
    print('File Read Success\n')
    return data_ref,data_tar
 

def PR(data_ref,data_tar):
    print("Start Ambiguity Function")
    t_axis = np.arange(0,(CIT*fs-1)/fs,1/fs)
    # array_Doppler_frequency = np.arange(-Doppler_frequency,Doppler_frequency,1/CIT)
    cols = int(CIT * fs)
    #矩阵的列数
    rows =int(len(data_ref) / cols)
    data_ref_array = data_ref[:int(rows*cols)].reshape(rows,cols)
    data_tar_array = data_tar[:int(rows*cols)].reshape(rows,cols)
    # print(data_ref[:9],'\n',data_ref_array[0,:9],'\n',data_ref_array.shape)
    # rows ,cols = data_ref.shape
    num_loop = len(array_Doppler_frequency)
    A_TD = np.zeros((rows,num_loop),'complex')
    for idx in tqdm(range(rows)):
        #print('处理时间点:',idx*CIT,'\n')
        temp_tar = data_tar_array[idx,:]
        temp_ref = data_ref_array[idx,:]
        #print(type(temp_tar),temp_tar.shape)
        for idx_fd in range(num_loop):
            f_d = array_Doppler_frequency[idx_fd]
            temp = np.multiply(temp_tar,np.conj(temp_ref))
            A_TD[idx,idx_fd] = np.dot(temp,np.exp(-1j*2*np.pi*f_d*t_axis))
    return A_TD

def plot_figure(A_TD):
    plot_TD = np.transpose(abs(A_TD))
    # print(plot_TD[:3,:3])
    row, col = plot_TD.shape
    for idx_col in range(col):
        plot_TD[:,idx_col] /= plot_TD[:,idx_col].max() 
        plot_TD[:,idx_col] = 20 * np.log10(plot_TD[:,idx_col])
    Threshold = -25
    plot_TD[plot_TD < Threshold] = Threshold
    x_step = 5
    y_step = 10
    xt = range(0,col,x_step)
    yt = range(0,row,y_step)
    x_label = array_start_time[:col][::x_step]
    y_label = array_Doppler_frequency[::y_step]
    fig = plt.figure(4,figsize=(20,12))
    im = plt.matshow(plot_TD, cmap='jet', \
        interpolation ='kaiser', aspect = 0.35)
    cb = plt.colorbar(im, fraction=0.045, pad=0.04, shrink=1.0)
    cb.set_label('Amplitude normalization (dB)',fontsize = 16)
    plt.tick_params(axis='both',which = 'major',labelsize = 14)
    plt.xticks(xt,x_label)
    plt.yticks(yt,y_label)
    plt.xlabel('Time (s)',fontsize = 20)
    plt.ylabel('Doppler Frequence (Hz)',fontsize = 20)
    plt.show()

def main():
    path = '/home/yuc/passive mmwave/eder_evk/RX/Dataset/data_0413/'
    file = 'x2_fast.dat'
    fname = path + file
    data_ref, data_tar = Read_file(fname)
    A_TD = PR(data_ref,data_tar)
    plot_figure(A_TD)
    
if __name__ == '__main__':
    main()
