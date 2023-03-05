#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on ...

@author: ...

Descripción
-----------

En este módulo podrías incluir las funciones más generales que quieras usar desde todos los TP's.

"""

    
def Estimador1(XX):
    
    temp = p.abs(XX[250,:])*2
    
    return temp

def Estimador2(XX, bins):
    
    Densidad_Pot= 2* np.abs(XX)**2
    Region_de_Pot = Densidad_Pot_rect[250-bins:250+bins+1, :]
    Pot_acotada_est = np.sum(Region_de_Pot_rect, axis = 0)
    
    return np.sqrt(2*Pot_total_est)

    
    

        