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
# import pdsmodulos asph pds

# def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
      
# tt= np.arange()

   
#     return (tt, xx)
vmax=1          #Amplitud Maxima [Volts]
dc=0            #Valor de continua [Volts]
ff=1 #Frecuencia en [Hz][]
ph=np.pi*1   #Fase [rad]
nn=8  #Muestras del ADC
fs=8#Frecuencia de muestreio del ADC [Hz]         
Ts=1/fs
delta_f=fs/nn


def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):

    tt = np.arange(0.0, nn/fs, 1/fs)
    aux = tt * 2*np.pi*ff

    xx = (np.sin(2*np.pi*ff*tt+ph))*vmax + dc

    return tt,xx

def mi_funcion_cos(vmax, dc, ff, ph, nn, fs):

    tt = np.arange(0.0, nn/fs, 1/fs)
    aux = tt * 2*np.pi*ff

    xx = (np.cos(2*np.pi*ff*tt+ph))*vmax + dc

    return tt,xx

def mi_funcion_step(vmax, dc, ff, ph, nn, fs):
    

    tt = np.arange(0.0, nn/fs, 1/fs)
    
    T=1/(2*np.pi*ff)
    
    N_delay= int(ph*T*fs/(2*np.pi))
    
  
    
    xx = np.zeros(N_delay)
    xx = np.append(xx, np.ones(nn-N_delay))
    # aux = np.ones(ph/)*-1
        
    xx = xx*vmax + dc

    return tt,xx

def mi_funcion_ramp(vmax, dc, ff, ph, nn, fs):
    

    tt = np.arange(0.0, nn/fs, 1/fs)
    
    T=1/(2*np.pi*ff)
    
    N_delay= int(ph*T*fs/(2*np.pi))
    
  
    
    xx = np.zeros(N_delay)
    
    indices= np.arange(0,nn-N_delay)
    
    aux = tt.take(indices)
    
    xx = np.append(xx, aux)
    xx = xx*vmax + dc

    return tt,xx

def W_twiddle(k,n,N):
    
    result=0
    result = np.round(np.cos(2*np.pi*k*n/N), 4)
    result -= np.round(np.sin(2*np.pi*k*n/N)*1j, 4)
  
    # print(result)
    
    return result

def DFT (xx):
    
    nn = np.size(xx)
    temp= np.zeros((nn,nn))
    temp= temp.astype(complex)
    
    XX = np.zeros(nn)
    XX = XX.astype(complex)
       
    for k in range(nn):
        for n in range(nn):
           if n==0:
               print("K= ",k)
           temp[k][n] = xx[n]*W_twiddle(k,n,nn)
           XX[k]+=temp[k][n]
           
           print("Actual", temp[k][n])
           print("Acumulado",XX[k])
       
    return np.round(XX,5), temp


            


Signal0 = mi_funcion_sen(vmax, dc, ff, ph*0, nn, fs)
# Signal1 = mi_funcion_cos(vmax, dc, fs, ph, nn, fs)

XX,xx = DFT(Signal0[1])
XX_abs=np.absolute(XX)
XX_ph=np.angle(XX,deg=True)
XX_df= np.arange(0.0, fs, fs/nn)

FFT2=np.round(np.fft.fft(Signal0[1]),7)
FFT=np.fft.fft(Signal0[1])

# Signal1 = mi_funcion_sen(vmax, dc, 1, ph*0.5, nn, fs)
# Signal1 = mi_funcion_ramp(vmax, dc, ff, ph*62, nn, fs)
# Signal2 = mi_funcion_ramp(vmax, dc, ff, ph*62, nn, fs)
# Signal4 = mi_funcion_step(vmax, dc, ff, ph*0, nn, fs)


# Para que funcione el qt -> %matplotlib qt5
plt.figure(1)
# plt.plot(Signal0[0], Signal0[1]-Signal1[1])
# plt.plot(Signal0[0], Signal0[1]-Signal1[1]-Signal2[1])
# plt.clf()
plt.plot(Signal0[0],Signal0[1], 'g:x')

# plt.plot(Signal1[0],Signal1[1], 'r:')
plt.xlabel('tiempo [s]')
plt.ylabel('Volt [V]')
plt.axis('tight')
plt.grid(which='both', axis='both')
plt.show()

plt.figure(2)
# plt.plot(Signal0[0], Signal0[1]-Signal1[1])
# plt.plot(Signal0[0], Signal0[1]-Signal1[1]-Signal2[1])
# plt.clf()
plt.plot(XX_df,XX_abs, 'cX')
# plt.plot(XX_df,XX_ph, 'rX')

# plt.plot(Signal1[0],Signal1[1], 'r:')
plt.xlabel('frecuencia [Hz]')
plt.ylabel('Volt [V]')
plt.axis('tight')
plt.grid(which='both', axis='both')
plt.show()

plt.figure(3)
# plt.plot(Signal0[0], Signal0[1]-Signal1[1])
# plt.plot(Signal0[0], Signal0[1]-Signal1[1]-Signal2[1])
# plt.clf()
# plt.plot(XX_df,XX_abs, 'bX')
plt.plot(XX_df,XX_ph, 'mX')

# plt.plot(Signal1[0],Signal1[1], 'r:')
plt.xlabel('frecuencia [Hz]')
plt.ylabel('Volt [V]')
plt.axis('tight')
plt.grid(which='both', axis='both')
plt.show()






    
