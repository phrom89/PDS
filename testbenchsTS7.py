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

from pandas import DataFrame
from IPython.display import HTML

# import basic_units as bu

# from basic_units import radians
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

# fr = np.random.rand(200, 1)

# noise_analog = np.random.normal(0,np.sqrt(pot_ruido), N_os)


# vmax=np.sqrt(2)

reali=200
amplitud=1
zero_padd = 0
Wbins=3

## El factor de proporcionalidad en la normalizacion es 
Omega0 = np.pi/2

fr = np.random.rand(1,reali).reshape(reali,1)
fr=fr-0.5
fr=(fr*4)*delta_f
# fr = fr - fr    #Esto lo hago para ver que pasaria si tengo el valor exacto siembre. Mato la dispercion.


Omega1 = (Omega0 + fr*((2*np.pi)/nn))

tt = np.arange(0.0, nn/fs, 1/fs)
tt = tt.reshape(1, tt.shape[0])
ff= np.arange(0.0, fs, fs/((zero_padd+1)*nn))
ff = ff.reshape(1,ff.shape[0])


arg = Omega1*fs*tt
xx_sen_mat=np.sin(arg)*amplitud
# xx_sen_mat=1




# xx_sen_mat = np.append(xx_sen_mat, np.zeros(zero_padd*nn))
# xx_sen_mat = xx_sen_mat.reshape(-1,1)


#Creo las distintas ventanas
win_Rectangular=sig.windows.boxcar(nn).reshape(1,nn)
win_Bartlett=np.bartlett(nn).reshape(1,nn)
win_Hann=np.hanning(nn).reshape(1,nn)
win_Blackman=np.blackman(nn).reshape(1,nn)
win_Flattop=sig.windows.flattop(nn).reshape(1,nn)

xx_rect=xx_sen_mat*win_Rectangular
xx_bart=xx_sen_mat*win_Bartlett
xx_hann=xx_sen_mat*win_Hann
xx_black=xx_sen_mat*win_Blackman
xx_flattop=xx_sen_mat*win_Flattop


xx_rect = np.append(xx_rect, np.zeros([reali,zero_padd*nn]),axis=-1)
xx_bart = np.append(xx_bart, np.zeros([reali,zero_padd*nn]),axis=-1)
xx_hann = np.append(xx_hann, np.zeros([reali,zero_padd*nn]),axis=-1)
xx_black = np.append(xx_black, np.zeros([reali,zero_padd*nn]),axis=-1)
xx_flattop = np.append(xx_flattop, np.zeros([reali,zero_padd*nn]),axis=-1)


# XX_sen_mat=fft(xx_sen_mat/xx_sen_mat.shape[1], axis=-1)

XX_rect = np.transpose(fft(xx_rect/xx_rect.shape[1], axis=-1))
XX_bart = np.transpose(fft(xx_bart/xx_bart.shape[1], axis=-1))
XX_hann = np.transpose(fft(xx_hann/xx_hann.shape[1], axis=-1))
XX_black = np.transpose(fft(xx_black/xx_black.shape[1], axis=-1))
XX_flattop = np.transpose(fft(xx_flattop/xx_flattop.shape[1], axis=-1))

# XX_3d = np.stack ((XX_rect, XX_bart, XX_hann, XX_black, XX_flattop))

# XX_sen_mat=fft(xx_sen_mat/xx_sen_mat.shape[1], axis=-1)




#%%  Calculo de Estimadores


#Estimador 1 es simplemente el modulo por dos.
def Estimador1(XX):
    
    temp = np.abs(XX[250,:])*2
    return temp

#Estimador2 calculamos la potencia y apartir de la potencia calculamos la amplitud.
def Estimador2(XX, bins):
    
    Densidad_Pot= 2* np.abs(XX)**2
    Region_de_Pot = Densidad_Pot[250-bins:250+bins+1, :]
    Pot_acotada_est = np.sum(Region_de_Pot, axis = 0)
    
    return np.sqrt(2*Pot_acotada_est)


# # El primer estimador, se calcula directamente como el valor absoluto al cuadrado.
# Est1amp_rect = np.abs(XX_rect[250,:])*2

# #Para el 2do estimador, tendremos que calcular la potencia, integrando una determinada region de la densidad de potencia.
# #Haremos un calculo de que porcentaje de la potencia total estamos usando al definir una determinada region.
# Densidad_Pot_rect = 2* np.abs(XX_rect)**2
# bfrec = ff < fs/2
# Pot_total_est = np.sum(Densidad_Pot_rect[bfrec], axis = 0)
# Region_de_Pot_rect = Densidad_Pot_rect[250-Wbins:250+Wbins+1, :]
# Pot_acotada_est = np.sum(Region_de_Pot_rect, axis = 0)
# Porcentaje_Pot_rect = (np.mean(Pot_acotada_est)/np.mean(Pot_total_est))*100
# print("Porcentaje de potencia" ,Porcentaje_Pot_rect)
# Est2amp_rect = np.sqrt(2*Pot_total_est)

#Armo una matriz con todos los estimadores para las distintas ventanas.
Estimadores = np.stack ((Estimador1(XX_rect), Estimador2(XX_rect, Wbins), Estimador1(XX_bart), Estimador2(XX_bart,Wbins), Estimador1(XX_hann), Estimador2(XX_hann,Wbins),Estimador1(XX_black), Estimador2(XX_black,Wbins),Estimador1(XX_flattop), Estimador2(XX_flattop,Wbins))).transpose()


#Calculo mediana, sesgo y varianza para todos los casos.
Medianas= np.median(Estimadores, axis = 0)
Sesgo= np.median(Estimadores, axis = 0) - amplitud
Varianza = np.mean((Estimadores - Medianas)**2, axis = 0)

#Armo mi matriz de resultados.
Resultados = np.stack ((Sesgo, Varianza)).transpose()



#Lo muestro en una tabla de valores
df = DataFrame(Resultados, columns=['$s_a$', '$v_a$'],
               index=[  
                        'Rectangular Slice',
                        'Rectangular Integral',
                        'Bartlett Slice',
                        'Bartlett Integral',
                        'Hann Slice',
                        'Hann Integral',
                        'Blackman Slice',
                        'Blackman Integral',
                        'Flat-top Slice',
                        'Flat-top Integral'
                     ])

#pandas.set_option('display.max_colwidth', 1)
HTML(df.to_html(col_space = '300px', justify = 'center'))
#HTML(df.to_html(notebook = True))





 


# est_amp = np.abs(XX[250, :])

# bfrec = ff1 <= fs

# ff= np.arange(0.0, fs, fs/((zero_padd+1)*nn))
# ff



# Pot_estimada= np.sum(2*np.abs(np.transpose(XX_sen_mat))**2, axis=0)

# sub_matriz= 


plt.figure(1)
plt.clf()

# Generar matriz de estimadores
# medianas=

# Sesgo = np.median(Estimadores) - amplitud 
# Varianza(np.mean())

rad=np.arange(0.0, 2*np.pi , (2*np.pi)/((zero_padd+1)*(nn)))

plt.plot( rad, 10* np.log10(2*np.abs((XX_rect))**2), lw=2,label='rect')
plt.plot( rad, 10* np.log10(2*np.abs((XX_bart))**2), lw=2,label='bart')
plt.plot( rad, 10* np.log10(2*np.abs((XX_hann))**2), lw=2,label='hann')
plt.plot( rad, 10* np.log10(2*np.abs((XX_black))**2), lw=2,label='black')
plt.plot( rad, 10* np.log10(2*np.abs((XX_flattop))**2), lw=2,label='flattop')

# plt.plot(tt, xx_rect[0,:])
# plt.plot(tt, xx_bart[0,:])

# plt.plot(tt, xx_black[0,:])

# plt.plot(tt, xx_hann[0,:])  
# plt.plot(tt, xx_flattop[0,:])


plt.figure(2)

plt.clf()

# plt.plot( rad, 2*np.abs((XX_rect))**2, lw=2,label='rect')
# plt.plot( rad, 2*np.abs((XX_bart))**2, lw=2,label='bart')
# plt.plot( rad, 2*np.abs((XX_hann))**2, lw=2,label='hann')
# plt.plot( rad, 2*np.abs((XX_black))**2, lw=2,label='black')
# plt.plot( rad, 2*np.abs((XX_flattop)), lw=2,label='flattop')

    # plt.plot( rad, 10* np.log10(2*np.abs((XX_3d[0,:,:]))**2), lw=2)

axes_hdl = plt.gca()
axes_hdl.legend()

plt.plot( rad, 10* np.log10(2*np.abs((XX_rect))**2), lw=2)
plt.plot( rad, 10* np.log10(2*np.abs((XX_bart))**2), lw=2)
plt.plot( rad, 10* np.log10(2*np.abs((XX_hann))**2), lw=2)
plt.plot( rad, 10* np.log10(2*np.abs((XX_black))**2), lw=2)
plt.plot( rad, 10* np.log10(2*np.abs((XX_flattop))**2), lw=2)



plt.figure(3)
plt.clf()

#Histogramas
plt.clf()
# plt.figure(1)
plt.title("Slice")
kwargs = dict(alpha=0.5,bins = 10, density=False, stacked=True)
kwargs_2 = dict(alpha=0.5, bins = 2,density=False, stacked=True)
plt.hist(Estimadores[:,0],**kwargs, label = "Rectangular")
plt.hist(Estimadores[:,2],**kwargs, label = "Bartlett")
plt.hist(Estimadores[:,4],**kwargs, label = "Hann")
plt.hist(Estimadores[:,6],**kwargs, label = "Blackman")
plt.hist(Estimadores[:,8],**kwargs, label = "FlatTop")
plt.legend()

plt.show()








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






    
