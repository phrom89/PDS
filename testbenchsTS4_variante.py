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
vmax=2          #Amplitud Maxima [Volts]
dc=0            #Valor de continua [Volts]
f0=1 #Frecuencia en [Hz][]
ph=np.pi*1   #Fase [rad]
nn=1000  #Muestras del ADC
fs=1000#Frecuencia de muestreio del ADC [Hz]
f0=fs/nn #Frecuencia en [Hz][]         
Ts=1/fs
delta_f=fs/nn
B_bits=4
vf=vmax
SNRa=60

# cantidad de veces más densa que se supone la grilla temporal para tiempo "continuo"
over_sampling = 4
N_os = nn*over_sampling
fs_os=fs*over_sampling

# datos del ruido
q=vf/2**(B_bits-1)
kn = 1
# pot_ruido = q**2/12 * kn # Watts (potencia de la señal 1 W)
pot_ruido = np.power(vf,2)/(2*np.power(10, SNRa/10))




def mi_funcion_sen(vmax, dc, ff, ph, nn, fs):

    tt = np.arange(0.0, nn/fs, 1/fs)
    # aux = tt * 2*np.pi*ff

    xx = (np.sin(2*np.pi*ff*tt+ph))*vmax + dc

    return tt,xx

def mi_funcion_cos(vmax, dc, ff, ph, nn, fs):

    tt = np.arange(0.0, nn/fs, 1/fs)
    # aux = tt * 2*np.pi*ff

    xx = (np.cos(2*np.pi*ff*tt+ph))*vmax + dc

    return tt,xx



def W_twiddle(k,n,N):
    
    result=0
    result = np.cos(2*np.pi*k*n/N)
    result -= np.sin(2*np.pi*k*n/N)*1j
  
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
           temp[k][n] = xx[n]*W_twiddle(k,n,nn)
           XX[k]+=temp[k][n]
           
           # print("Actual", temp[k][n])
           # print("Acumulado",XX[k])
       
    return XX

def Cuantizar (xx, vf, bits):   #Revisar hay algo mal
    
    nn=np.size(xx)
    x = np.zeros(nn)
    delta_q= vf/2**(bits-1)
    # lim_pos = delta_q*((2**(bits-1)-1))
    lim_pos = delta_q*((2**(bits-1)))
    lim_neg = 0-lim_pos
    
    for n in range(nn):
        
        
        x[n]=xx[n]/delta_q
        x[n]=np.round(x[n])
        x[n]=x[n]*delta_q
        if x[n] >= lim_pos:
            x[n]=lim_pos
        elif x[n] <= lim_neg:
            x[n]=lim_neg
            
         
    return x

def discretizar (xx, ov, nn, fs):
    
    x_d = np.zeros(nn)
    tt = np.arange(0.0, nn/fs, 1/fs)
      
    for n in range(nn):
          
        x_d[n]=xx[ov*n]
          
    return tt,x_d


# Funcion para tiempo "continuo"
tt_os, xx_analog = mi_funcion_sen(vmax, dc, f0, ph*0, N_os, fs_os)


# Funcion a la que se le sumara ruido y se va a cuantizar.
# tt, xx = mi_funcion_sen(vmax, dc, f0, ph*0, nn, fs)

#Calculo ruido y lo sumo a la señal
noise_analog = np.random.normal(0,np.sqrt(pot_ruido), N_os)
xx_sn_analog = noise_analog + xx_analog

tt = tt_os[::over_sampling]
xx_d = xx_sn_analog[::over_sampling]



xx_q= Cuantizar(xx_d, vf, B_bits)

noise_digital=xx_q-xx_d

error_mean=np.mean(noise_digital)
error_var=np.var(noise_digital)
error_ac = sig.correlate( noise_digital, noise_digital)


ff= np.arange(0.0, fs, fs/nn)
ff_os= np.arange(0.0, fs_os, fs_os/ N_os )

ft_xx_analog=fft(xx_analog)
ft_xx=fft(xx_d)
ft_xx_q=fft(xx_q)

N_analog=fft(noise_analog)
N_digital=fft(noise_digital)


cte_norm_os  = np.amax(10* np.log10((2*np.abs(ft_xx_analog)**2)/N_os)) 
db_norm_os = np.full(N_os, cte_norm_os)

cte_norm_q  = np.amax(10* np.log10((2*np.abs(ft_xx_q)**2)/nn)) 
db_norm_q = np.full(nn, cte_norm_q)

bfrec = ff <= fs/2
bfrec_os = ff_os <= fs/2


# print('Media teorica: 0                     Estimación de la media: {:g}'.format(error_mean) )
print('SNRa: {:g} '.format(np.log10(np.var(xx_analog)/np.var(noise_analog))*10))






# XX_FFT=fft(Signal0[1])

# Signal1 = mi_funcion_sen(vmax, dc, 1, ph*0.5, nn, fs)
# Signal1 = mi_funcion_ramp(vmax, dc, f0, ph*62, nn, fs)
# Signal2 = mi_funcion_ramp(vmax, dc, f0, ph*62, nn, fs)
# Signal4 = mi_funcion_step(vmax, dc, f0, ph*0, nn, fs)


# Para que funcione el qt -> %matplotlib qt5
# plt.close("all")
# plt.figure(1)

# # plt.plot(Signal0[0],xx_q, 'g:x')
# # plt.plot(Signal0[0],xx, 'b:+')
# # plt.plot(Signal0[0],error, 'r:x')

# # plt.plot(Signal1[0],Signal1[1], 'r:')
# plt.xlabel('tiempo [s]')
# plt.ylabel('Volt [V]')
# plt.axis('tight')
# plt.grid(which='both', axis='both')
# # plt.show()

plt.figure(1)
plt.clf()
plt.plot(tt, xx_q, lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)')
plt.plot(tt, xx_d, linestyle=':', color='green',marker='o', markersize=3, markerfacecolor='none', markeredgecolor='green', fillstyle='none', label='$ s_R = s + n $  (ADC in)')
plt.plot(tt_os, xx_analog, color='orange', ls='dotted', label='$ s $ (analog)')
 
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B_bits, vf, q) )
plt.xlabel('tiempo [segundos]')
plt.ylabel('Amplitud [V]')
axes_hdl = plt.gca()
axes_hdl.legend()
plt.show()




# ft_xx_analog=fft(xx_analog) Señal continua
# ft_xx=fft(xx_d)  Señal discretizada.. continua + ruido
# ft_xx_q=fft(xx_q) Señal cuantizada.

# N_analog=fft(noise_analog)
# N_digital=fft(noise_digital)



plt.figure(2)
plt.clf()
bfrec = ff <= fs/2
 
Nnq_mean = np.mean(np.abs(N_digital)**2)*1/nn
nNn_mean = np.mean(np.abs(np.sqrt(1/N_os)*N_analog)**2)

 
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(np.sqrt(1/nn)*ft_xx_q[bfrec])**2) - db_norm_q[bfrec], lw=2, label='$ s_Q = Q_{B,V_F}\{s_R\} $ (ADC out)' )
# plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(2*np.abs(np.sqrt(1/N_os)*ft_xx_analog[ff_os <= fs/2])**2) , color='orange', ls='dotted', label='$ s $ (analog)' )
# plt.plot( ff[bfrec], 10* np.log10(2*np.abs(np.sqrt(1/nn)*ft_xx[bfrec])**2), ':g', label='$ s_R = s + n $  (ADC in)' )

plt.plot( ff_os[ff_os <= fs/2], 10* np.log10(np.abs(N_analog[ff_os <= fs/2])**2/np.abs(ft_xx_analog[1])**2), ':r')
plt.plot( ff[bfrec], 10* np.log10(np.abs(np.sqrt(2/nn)*N_digital[bfrec]/nn)**2), ':c')
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([nNn_mean, nNn_mean]) ), '--r', label= '$ \overline{n} = $' + '{:3.1f} dB (piso analog.)'.format(10* np.log10(2* nNn_mean)) )
plt.plot( np.array([ ff[bfrec][0], ff[bfrec][-1] ]), 10* np.log10(2* np.array([Nnq_mean, Nnq_mean]) ), '--c', label='$ \overline{n_Q} = $' + '{:3.1f} dB (piso digital)'.format(10* np.log10(2* Nnq_mean)) )
plt.title('Señal muestreada por un ADC de {:d} bits - $\pm V_R= $ {:3.1f} V - q = {:3.3f} V'.format(B_bits, vf, q) )
plt.ylabel('Densidad de Potencia [dB]')
plt.xlabel('Frecuencia [Hz]')
axes_hdl = plt.gca()
axes_hdl.legend()
# suponiendo valores negativos de potencia ruido en dB
# plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))
# plt.ylim(-80,80)

# plt.ylim((1.5*np.min(10* np.log10(2* np.array([Nnq_mean, nNn_mean]))),10))


# plt.figure(3)
# plt.clf()
# plt.plot(error_ac)
# plt.title( 'Secuencia de autocorrelación {:d} bits - VRef={:3.1f} V'.format(B_bits,vf))
# plt.ylabel('Autocorrelacion [#]')
# plt.xlabel('Demora [#]')
# plt.show()



# plt.figure(2)
# # plt.plot(Signal0[0], Signal0[1]-Signal1[1])
# # plt.plot(Signal0[0], Signal0[1]-Signal1[1]-Signal2[1])
# # plt.clf()
# plt.plot(XX_df,XX_abs, 'cX')
# # plt.plot(XX_df,XX_ph, 'rX')

# # plt.plot(Signal1[0],Signal1[1], 'r:')
# plt.xlabel('frecuencia [Hz]')
# plt.ylabel('Volt [V]')
# plt.axis('tight')
# plt.grid(which='both', axis='both')
# plt.show()

# plt.figure(3)
# # plt.plot(Signal0[0], Signal0[1]-Signal1[1])
# # plt.plot(Signal0[0], Signal0[1]-Signal1[1]-Signal2[1])
# # plt.clf()
# # plt.plot(XX_df,XX_abs, 'bX')
# plt.plot(XX_df,XX_ph, 'mX')

# # plt.plot(Signal1[0],Signal1[1], 'r:')
# plt.xlabel('frecuencia [Hz]')
# plt.ylabel('Volt [V]')
# plt.axis('tight')
# plt.grid(which='both', axis='both')
plt.show()






    
