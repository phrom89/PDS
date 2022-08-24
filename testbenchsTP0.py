#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 19:39:26 2022

Primeros pasos en la creacion de funciones en python.
Vamos a jugar con la libreria numpy para generar distintas señales variando sus parametros

@author: promero
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import pdsmodulos as pds

# def mi_funcion_sen (vmax, dc, ff, ph, nn, fs):
      
# tt= np.arange()

   
#     return (tt, xx)
vmax=3          #Amplitud Maxima [Volts]
dc=1            #Valor de continua [Volts]
ff=10           #Frecuencia en [Hz][]
ph=np.pi*0      #Fase [rad]
nn=1000         #Muestras del ADC
fs=1000         #Frecuencia de muestreio del ADC [Hz]         
# Ts=1/Fs


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




Signal = mi_funcion_sen(vmax, dc, ff, ph, nn, fs)
# Signal = mi_funcion_cos(vmax, dc, ff, ph, nn, fs)

plt.plot(Signal[0], Signal[1])
plt.xlabel('tiempo [s]')
plt.ylabel('Volt [V]')
plt.axis('tight')
plt.show()





    
