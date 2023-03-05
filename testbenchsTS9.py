#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:39:26 2022

Primeros pasos en la creacion de funciones en python.
Vamos a jugar con la libreria numpy para generar distintas señales variando sus parametros
Señales logradas: sen(),cos(),step() y ramp().
Se trataron de hacer aperaciones entre señales de manera muy rudimentaria. Algo se logró.

@author: promero
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.fft import fft
import scipy.io as sio
import scipy.signal as sig
# import pdsmodulos asph pds

# def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
      
# tt= np.arange()

   
#     return (tt, xx)
vmax=1       #Amplitud Maxima [Volts]
dc=0            #Valor de continua [Volts]
f0=1 #Frecuencia en [Hz][]
ph=np.pi*1   #Fase [rad]
nn=1000  #Muestras del ADC
fs=1000#Frecuencia de muestreio del ADC [Hz]
k0=nn/4
f0=k0*fs/nn #Frecuencia en [Hz][]         
Ts=1/fs
delta_f=fs/nn
B_bits=4
vf=2
SNRa=25


# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 4
N_os = nn*over_sampling
fs_os=fs*over_sampling

# datos del ruido
q=vf/2**(B_bits-1)


pot_ruido = np.power(vmax,2)/(2*np.power(10, SNRa/10))

kn = 1
# pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)

block = 1024 #760

mat_struct = sio.loadmat('ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead']
fraccion_ECG= ecg_one_lead[0:100000]


fff, Pxx_den = sig.welch(fraccion_ECG,fs=fs, nperseg=block, axis = 0 )



qrs = mat_struct['qrs_detections']


plt.figure(0)
plt.clf()


plt.plot(fff, Pxx_den)



plt.show()


# index_energia = np.where(np.cumsum.)


# plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_bart))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_hann))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_black))**2), lw=2)




# # fr = np.random.rand(200, 1)

# # noise_analog = np.random.normal(0,np.sqrt(pot_ruido), N_os)


# # vmax=np.sqrt(2)
# zero_padd = 0
# reali=10
# amplitud=2


# Omega0 = np.pi/2

# fr = np.random.rand(reali, 1)
# fr=fr-0.5
# fr=fr*4

# Omega1 = (Omega0 + fr*((2*np.pi)/nn))

# tt = np.arange(0.0, nn/fs, 1/fs)

# arg= Omega1*fs*tt
# xx_sen_mat=np.sin(arg)*amplitud

# #Creo las distintas ventanas
# win_Rectangular=sig.windows.boxcar(nn)
# win_Bartlett=np.bartlett(nn)
# win_Hann=np.hanning(nn)
# win_Blackman=np.blackman(nn)
# win_Flattop=sig.windows.flattop(nn)

# xx_rect=xx_sen_mat*win_Rectangular
# xx_bart=xx_sen_mat*win_Bartlett
# xx_hann=xx_sen_mat*win_Hann
# xx_black=xx_sen_mat*win_Blackman
# xx_flattop=xx_sen_mat*win_Flattop



# XX_sen_mat=fft(xx_sen_mat/xx_sen_mat.shape[1], axis=-1)

# XX_rect = np.transpose(fft(xx_rect/xx_rect.shape[1], axis=-1))
# XX_bart = np.transpose(fft(xx_bart/xx_bart.shape[1], axis=-1))
# XX_hann = np.transpose(fft(xx_hann/xx_hann.shape[1], axis=-1))
# XX_black = np.transpose(fft(xx_black/xx_black.shape[1], axis=-1))
# XX_flattop = np.transpose(fft(xx_flattop/xx_flattop.shape[1], axis=-1))

# XX_3d = np.stack ((XX_rect, XX_bart, XX_hann, XX_black, XX_flattop))

# XX_sen_mat=fft(xx_sen_mat/xx_sen_mat.shape[1], axis=-1)   


# rad=np.arange(0.0, 2*np.pi, (2*np.pi)/((zero_padd+1)*nn))


# # est_amp = np.abs(XX[250, :])

# # bfrec = ff1 <= fs

# # ff= np.arange(0.0, fs, fs/((zero_padd+1)*nn))
# # ff



# # Pot_estimada= np.sum(2*np.abs(np.transpose(XX_sen_mat))**2, axis=0)

# # sub_matriz= 


# plt.figure(0)
# plt.clf()

# # Generar matriz de estimadores
# # medianas=

# # Sesgo = np.median(Estimadores) - amplitud 
# # Varianza(np.mean())



# plt.plot( rad, 10* np.log10(2*np.abs((XX_3d[4,:,:]))**2), lw=2)
# plt.plot( rad, 10* np.log10(2*np.abs((XX_3d[0,:,:]))**2), lw=2)

# # axes_hdl = plt.gca()
# # axes_hdl.legend()

# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_rect))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_bart))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_hann))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_black))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_flattop))**2), lw=2)



# plt.show()








# # tt, xx_sen1 = mi_funcion_sen(vmax, dc, f0, ph*0, nn, fs)
# # tt, xx_sen2 = mi_funcion_sen(vmax, dc, f0+0.1, ph*0, nn, fs)
# # tt, xx_sen3 = mi_funcion_sen(vmax, dc, f0+0.25, ph*0, nn, fs)
# # tt, xx_sen4 = mi_funcion_sen(vmax, dc, f0+0.5, ph*0, nn, fs)

# # xx_sen1=xx_sen1/np.sqrt(np.var(xx_sen1))
# # xx_sen2=xx_sen2/np.sqrt(np.var(xx_sen2))
# # xx_sen3=xx_sen3/np.sqrt(np.var(xx_sen3))
# # xx_sen4=xx_sen4/np.sqrt(np.var(xx_sen4))

# # zero_padd = 10

# # xx_sen1 = np.append(xx_sen1, np.zeros(zero_padd*nn))
# # xx_sen2 = np.append(xx_sen2, np.zeros(zero_padd*nn)) 
# # xx_sen3 = np.append(xx_sen3, np.zeros(zero_padd*nn)) 
# # xx_sen4 = np.append(xx_sen4, np.zeros(zero_padd*nn))  

# # XX_sen1=fft(xx_sen1)/xx_sen1.shape[0]
# # XX_sen2=fft(xx_sen2)/xx_sen2.shape[0]
# # XX_sen3=fft(xx_sen3)/xx_sen3.shape[0]
# # XX_sen4=fft(xx_sen4)/xx_sen4.shape[0]

# # ff1= np.arange(0.0, fs, fs/((zero_padd+1)*nn))
# # # ff2= np.arange(0.0, fs, fs/nn)

# # bfrec = ff1 <= fs

# # Area1= np.sum(2*np.abs(XX_sen1[bfrec])**2)
# # Area2= np.sum(2*np.abs(XX_sen2[bfrec])**2)
# # Area3= np.sum(2*np.abs(XX_sen3[bfrec])**2)
# # Area4= np.sum(2*np.abs(XX_sen4[bfrec])**2)


# # plt.figure(1)
# # plt.clf()

 
# # plt.plot( ff1[bfrec], (2*np.abs(XX_sen1[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0, Area1))
# # plt.plot( ff1[bfrec], (2*np.abs(XX_sen2[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.1, Area2))
# # plt.plot( ff1[bfrec], (2*np.abs(XX_sen3[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.25, Area3))
# # plt.plot( ff1[bfrec], (2*np.abs(XX_sen4[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.5, Area4))
# # # plt.plot(tt, xx_d, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
# # axes_hdl = plt.gca()
# # axes_hdl.legend()


# # plt.show()

# # plt.figure(2)
# # plt.clf()

# # plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen1[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0, Area1))
# # plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen2[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.25, Area2))
# # plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen3[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.5, Area3))
# # plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen4[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.75, Area4))
# # # plt.plot(tt, xx_d, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
# # axes_hdl = plt.gca()
# # axes_hdl.legend()


# # plt.show()






    
