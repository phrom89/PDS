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





def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):

    tt = np.arange(0.0, nn/fs, 1/fs)
    # aux = tt * 2*np.pi*ff

    xx = (np.sin(2*np.pi*ff*tt+ph))*vmax + dc

    return tt,xx



# vmax=np.sqrt(2)
tt, xx_sen1 = mi_funcion_sen(vmax, dc, f0, ph*0, nn, fs)
tt, xx_sen2 = mi_funcion_sen(vmax, dc, f0+0.25, ph*0, nn, fs)
tt, xx_sen3 = mi_funcion_sen(vmax, dc, f0+0.5, ph*0, nn, fs)
tt, xx_sen4 = mi_funcion_sen(vmax, dc, f0+0.75, ph*0, nn, fs)

xx_sen1=xx_sen1/np.sqrt(np.var(xx_sen1))
xx_sen2=xx_sen2/np.sqrt(np.var(xx_sen2))
xx_sen3=xx_sen3/np.sqrt(np.var(xx_sen3))
xx_sen4=xx_sen4/np.sqrt(np.var(xx_sen4))



zero_padd = 9
xx_sen1 = np.append(xx_sen1, np.zeros(zero_padd*nn))
xx_sen2 = np.append(xx_sen2, np.zeros(zero_padd*nn)) 
xx_sen3 = np.append(xx_sen3, np.zeros(zero_padd*nn)) 
xx_sen4 = np.append(xx_sen4, np.zeros(zero_padd*nn))  

XX_sen1=fft(xx_sen1)/xx_sen1.shape[0]
XX_sen2=fft(xx_sen2)/xx_sen2.shape[0]
XX_sen3=fft(xx_sen3)/xx_sen3.shape[0]
XX_sen4=fft(xx_sen4)/xx_sen4.shape[0]

ff1= np.arange(0.0, fs, fs/((zero_padd+1)*nn))
# ff2= np.arange(0.0, fs, fs/nn)

bfrec = ff1 <= fs

Area1= np.sum(2*np.abs(XX_sen1[bfrec])**2)
Area2= np.sum(2*np.abs(XX_sen2[bfrec])**2)
Area3= np.sum(2*np.abs(XX_sen3[bfrec])**2)
Area4= np.sum(2*np.abs(XX_sen4[bfrec])**2)


plt.figure(1)
plt.clf()

 
plt.plot( ff1[bfrec], (2*np.abs(XX_sen1[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0, Area1))
plt.plot( ff1[bfrec], (2*np.abs(XX_sen2[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.25, Area2))
plt.plot( ff1[bfrec], (2*np.abs(XX_sen3[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.5, Area3))
plt.plot( ff1[bfrec], (2*np.abs(XX_sen4[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.75, Area4))
# plt.plot(tt, xx_d, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
axes_hdl = plt.gca()
axes_hdl.legend()


plt.show()

plt.figure(2)
plt.clf()

plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen1[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0, Area1))
plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen2[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.25, Area2))
plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen3[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.5, Area3))
plt.plot( ff1[bfrec], 10* np.log10(2*np.abs(XX_sen4[bfrec])**2), lw=2, label='freq= {:3.3f} Area= {:3.3f}'.format(f0+0.75, Area4))
# plt.plot(tt, xx_d, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
axes_hdl = plt.gca()
axes_hdl.legend()



plt.show()






    
