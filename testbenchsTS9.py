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

block = 4000
# block = 4000
# block = 2000

mat_struct = sio.loadmat('ECG_TP4.mat')
ecg_one_lead = mat_struct['ecg_lead']


#Analizando la señal ecg completa, identificamos bloques relacionados al
#paciente en reposo,donde el perido es mayor, en ejercico, donde el periodo
#aumenta y el pico de exigencia, donde el periodo es máximo. 

ecg_reposo= ecg_one_lead[0:100000]
ecg_ejercicio= ecg_one_lead[450000:550000]
ecg_pico= ecg_one_lead[750000:850000]


#Calculamos la densidad espectral de potencia para cada caso.
ff, Pxx_reposo = sig.welch(ecg_reposo,fs=fs, nperseg=block, axis = 0 )
ff, Pxx_ejercicio = sig.welch(ecg_ejercicio,fs=fs, nperseg=block, axis = 0 )
ff, Pxx_pico = sig.welch(ecg_pico,fs=fs, nperseg=block, axis = 0 )


#Buscamos la frecuencia de corte para un bw que represente el 90%, 95% y 99% de potencia de la señal
#para los tres casos.

bw90 = 0.90
bw95 = 0.95
bw99 = 0.99

index_Energia = np.where(np.cumsum(Pxx_reposo)/np.sum(Pxx_reposo) > bw90)[0]
W_corte90rep = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_reposo)/np.sum(Pxx_reposo) > bw95)[0]
W_corte95rep = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_reposo)/np.sum(Pxx_reposo) > bw99)[0]
W_corte99rep = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_ejercicio)/np.sum(Pxx_ejercicio) > bw90)[0]
W_corte90ej = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_ejercicio)/np.sum(Pxx_ejercicio) > bw95)[0]
W_corte95ej = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_ejercicio)/np.sum(Pxx_ejercicio) > bw99)[0]
W_corte99ej = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_pico)/np.sum(Pxx_pico) > bw90)[0]
W_corte90pic = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_pico)/np.sum(Pxx_pico) > bw95)[0]
W_corte95pic = ff[index_Energia[0]]

index_Energia = np.where(np.cumsum(Pxx_pico)/np.sum(Pxx_pico) > bw99)[0]
W_corte99pic = ff[index_Energia[0]]






plt.figure(0)
plt.clf()


plt.plot(ff, Pxx_reposo, label = 'reposo')
plt.plot(ff, Pxx_ejercicio, label = 'ejercicio')
plt.plot(ff, Pxx_pico, label = 'pico')
plt.legend()
plt.xlim(0,10)

plt.show()

plt.figure(1)
plt.clf()


plt.plot(ff, 10*np.log10(Pxx_reposo), label = 'reposo')
plt.plot(ff, 10*np.log10(Pxx_ejercicio), label = 'ejercicio')
plt.plot(ff, 10*np.log10(Pxx_pico), label = 'pico')
plt.xlabel("Frecuencia [Hz]")
plt.ylabel('Potencia [dB]')
plt.xlim([0,200])
plt.grid()
plt.legend()

plt.figure(2)
plt.clf()

plt.plot(ecg_reposo[3000:4500], label = 'reposo')
plt.plot(ecg_ejercicio[3000:4500], label = 'ejercicio')
plt.plot(ecg_pico[3000:4500], label = 'pico')

#De medir la distancia entre picos, sacamos que la frecuencia cardiaca en cada caso esta en el orden de
#1,32HZ para reposo, 1,89Hz para ejercicio y 2,26Hz para pico.

plt.legend()

plt.show()

# index_energia = np.where(np.cumsum.)


# plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_bart))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_hann))**2), lw=2)
# # plt.plot( rad, 10* np.log10(2*np.abs(np.transpose(XX_black))**2), lw=2)


qrs_detections = mat_struct['qrs_detections']


# Viendo el grafico podemos ver que el latido va desde 0 a 600
# qsr me da el pico del latido que se encuentra en la primera muestra en 250
# por eso si me quiero quedar con toda la informacion desde el ecg selecciono desde
# (pico - 250) hasta (pico + 350)
inferior = 250
sup = 350

latido = (ecg_one_lead[int(qrs_detections[0] - inferior):int(qrs_detections[0] + sup)])
realizaciones = np.arange(len(qrs_detections))
latidos = np.zeros([sup+inferior, qrs_detections.shape[0]])

for i in realizaciones:
    latidos[:,i] = ecg_one_lead[int(qrs_detections[i] - inferior):int(qrs_detections[i] + sup)].flatten()
    latidos[:,i]  -= np.mean(latidos[:,i]) # le resto su valor medio para centrarlos


plt.figure(3)
plt.clf()

plt.plot(latidos)
plt.autoscale(enable=True, axis='x', tight=True)
plt.title("Latidos")
plt.legend()
plt.show()


#Buscamos un espacio donde este bien definido cuales son normales y cuales ventriculares 
#Lo encontramos en 241, 111000
slicing_latidos = latidos[241, :]

# Los que estan por debajo de 11500 son latidos normales
# Caso contrario pertenecen a la categoria de ventriculares
filtro_normal = slicing_latidos < 11100 #vector booleano
filtro_ventricular = slicing_latidos > 11100 #vector booleano




plt.figure(4)
plt.clf()

plt.plot(latidos[:,filtro_normal], 'b',alpha = 0.5, linewidth=3.0)
plt.plot(latidos[:,filtro_ventricular], 'g', alpha = 0.5,  linewidth=3.0)
plt.grid()
plt.title("Latidos presentes en el registro agrupados por tipo")
plt.xlabel('Tiempo [mSeg]')
plt.ylabel('Amplitud')

#Por ultimo buscamos graficar una latido ventricular y uno normal que representa la media de
#lo que venimos haciendo

plt.figure(5)
plt.clf()

lat_vent = np.mean(latidos[:,filtro_ventricular], axis = 1)
lat_norm = np.mean(latidos[:,filtro_normal], axis = 1)

plt.plot(lat_vent, 'b', label = 'Ventricular',alpha = 0.5, linewidth=3.0)
plt.plot(lat_norm, 'g', label = 'Normal', alpha = 0.5,  linewidth=3.0)
plt.legend()
plt.grid()


plt.show()









    
