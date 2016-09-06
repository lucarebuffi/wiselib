# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 16:41:12 2016

@author: Mic
"""

#%%

from __future__ import division
from numpy import *
import numpy as np
import wise.Rayman5 as rm
from  wise.Rayman5 import Amp, Cyc
import wise.Optics as Optics
import multiprocessing
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot

#%%
if __name__ == '__main__':

    matplotlib.rcParams['figure.max_open_warning'] = '0'

#%% 1 - Initialization of optics (wavelength, source, mirror, etc)

    # 0) parametri generali
    #------------------------------------------------------------
    Det_Size = 50e-6  # detector physical size (m)
    NPools = 0			# N of parallel pools (used for focussing).0 = no multiprocessing
    PathFigError = "data/Norm_H19jan11.txt" # figure error
    PathPsd = "data/psd_00settete.dat"

    # 2) Pupolo KB
    #------------------------------------------------------------
    L = 0.4				# Kb physical size (m)
    kbv = Optics.Ellipse(f1 = 98., f2 = 1.2, Alpha = 2*pi/180, L = L)
    '''
    Come dicevo, Ellipse per ora NON accetta il posizionamento (XOrigin, YOrigin, Theta)
    '''
    #	2a) Figure error (si potrebbe fare la funzione FigureErrorLoad)
    #------------------------------------------------------------
    FigError = np.loadtxt(PathFigError ) * 1
    kbv.FigureErrorAdd(FigError * 1e-3, 2e-3) # (m)

    # 1) Pupolo Sorgente (ancora parziale)
    #------------------------------------------------------------
    Lambda = 10e-9    # wavelength
    Waist0 = 125e-6 * sqrt(2) ;
    Sg = Optics.GaussianSource_1d(Lambda, Waist0 ,ZOrigin = kbv.XYF1[0], YOrigin =  kbv.XYF1[1], Theta = kbv.p1_Angle)
    '''
     note: qui la sorgente usa dati dello specchio (che è dopo), e risponde alla domanda "fatemi mettere la sorgente nella focale dello specchio". Bisognerebbe forse introdurre le variabili ZOrigin, YOrigin e Theta nella classe Ellipse, in modo da "scaricare" la responsabilità dell'orientamento agli elementi ottici e non alla sorgente.
    Accordarsi: definire il sistema di assi XYZ. Il nostro stesso codice al momento è NON uniforme, ma facilmente correggibile. Nella classe GaussianSource_1d Z è la direzione di propagazione e Y la coordinata trasversa. Nelle coppie di numeri (eg ZYOrigin) faccio comparire sempre prima la Z (che mimerebbe la X) e per seconda la Y.
    Similmente nella classe Ellipse, la X è la direzione di propagazione e la Y è sempre la direzione trasversa, e faccio comparire XOrigin, YOrigin o XYOrigin.
    C'è nell'aria di correggere la Z del fascio gaussiano in X, ma siamo ottici e la direzione di propagazione è la Z :-)
    '''


    # 	2b) Roughness
    #------------------------------------------------------------
    '''
    	 Qui tutti i numeri sono in S.I.
    	 se il file è in um e nm^3 allora devo mettere scaling
    	 1e-6 e (1e-27)
    	 se non funziona, si può riscrivere tutto in mm
    '''
    kbv.Roughness.NumericPsdLoadXY(PathPsd,xScaling = 1e-6,
    											yScaling = 1e-27,																			xIsSpatialFreq = False)
    kbv.Roughness.Options.FIT_NUMERIC_DATA_WITH_POWER_LAW = False
    kbv.Options.USE_ROUGHNESS = False #<cambia qui>


    #%% 2 - Focussing (script)
    '''
    Il codice qui è in forma "non oasis", ovviamente. Vengono usate le funzioni elementari concepite nei diversi oggetti. Alcune cose ci è già chiaro come dovranno essere messe in forma oasis, altre no.
    '''
    t0 = time.time()
    # i) Auto Sampling (easy way)
    # Info da: Fascio(Lambda), piano kb e piano detector
    # cf commenti dopo
    Theta0 = kbv.pTan_Angle
    Theta1 =  arctan(-1/kbv.p2[0])
    L0 = kbv.L
    L1 = Det_Size
    z = kbv.f2
    NAuto = rm.SamplingCalculator(Sg.Lambda, kbv.f2, kbv.L, Det_Size, Theta0, Theta1)

    # ii) Piano specchio (Sorgente=>Specchio)
    Mir_x, Mir_y = kbv.GetXY_MeasuredMirror(NAuto,0)
    Mir_E = Sg.EvalField_XYLab(Mir_x, Mir_y)


    # ii ) Defocus sweep
    # cf commenti dopo
    #<cambia qui>
    DefocusList = arange(-10e-3, 10e-3, 1e-3)
    #DefocusList = [0]
    NDefocus = len(DefocusList)
    E1List = np.empty((NDefocus, NAuto) , dtype = complex)
    HewList = np.zeros(NDefocus)
    for i, Defocus in enumerate(DefocusList):
    	print ('Processing %d/%d: Defocus = %0.1f mm' %(i, NDefocus, (Defocus * 1e3)))
    	# Specchio => Detector
    	Det_x, Det_y = kbv.GetXY_TransversePlaneAtF2(Det_Size, NAuto, Defocus )
    	Det_ds = np.sqrt((Det_x[0] - Det_x[-1])**2 + (Det_y[0] - Det_y[-1])**2)
    	# E1
    	E1List[i,:] = rm.HuygensIntegral_1d_MultiPool(Lambda,
    											Mir_E,Mir_x, Mir_y, Det_x, Det_y, NPools)
    	HewList[i] = rm.HalfEnergyWidth_1d(abs(E1List[i,:])**2, Step = Det_ds)
    	plt.figure()
    	plt.suptitle('DeltaZ = %0.2f' %(Defocus * 1e3))
    	plt.plot(abs(E1List[i,:])**2)

    I1List = abs(E1List)**2
    Mir_s = rm.xy_to_s(Mir_x, Mir_y)
    Det_s = rm.xy_to_s(Det_x, Det_y)

    t1 = time.time()

    print('elapsed time: %s' % str(t1-t0))
    #%% Come si potrebbe scrivere
    '''
    nota1: In alcuni casi, i parametri di un elemento ottico possono dipendere dal precedente . P.e: il detector, che si vuole mettere nel fuoco dello specchio. Chiedere a Rebuffi se Oasis fa già questo

    A = Source(Lambda, Angolo, altri parametri)

    B = Ellipse(distanza da sorgente, altri parametri)

    C = MakePlane(distanza suggerita da elemento ottico precedente + delta utente, angolo suggerito da elemento ottico precedente + delta utente)

    E_su_C = C.GetField
    poi C invocherà B.GetField (numerico) che invocherò A.GetField (analitico)

    Il calcolo automatico del sampling è una cosa delicata:
    Dati due piani, dipende (a meno di altri fattori) dal prodotto delle lunghezze fisiche dei due: L1*L2
    Per più piani, logica vorrebbe che il campionamento fosse uguale per tutti e pari a max{L0*L1, L1*L2, L2*L3}. Tuttavia la cosa è di difficile concezione, e almeno in prima battuta conviene determinare il campionamento a coppie: N1 = L1*L0, N2= L2*L1, etc. Questo rende la propagazione intrinsecamente una funzione a due elementi, infatti per passare da 0 a 1 serve N0 = N1  = L1*L2. Poi per passare da 1 a 2 serve N1'=N2'. Se N1' != N1 allora bisognerà binnare o interpolare il campo. Poi da 2 a 3 si userà N2'' = N3'' e così via.

    nota2:
    	lo sweep in z dello script è in realtà un uso particolare e serve per trovare il fuoco migliore (dato dal minimo della HEW). Di fatto è quello che si potrebbe chiamare un python script.
    '''


    #%% PLOT

    # E0 (Mir_E, field on the mirror)
    plt.suptitle('|E0|: specchio ')
    plot(Mir_s *1000, Amp(Mir_E),'r')
    plt.xlabel('rho (mm)')

    # plot cicli ottici
    plt.figure(200)
    plot(Mir_s * 1e3, Cyc(Mir_E),'g')
    plt.xlabel('mm')

    #	A1 = abs(E1) / max(abs(E1))
    #	N2 = int(floor(len(Mir_x)/2))
    Det_s = rm.xy_to_s(Det_x, Det_y)

    # E1 (Det_E, field on the detector)
    plt.figure(210)
    plt.suptitle('E1 intensity focalizzato')
    plt.xlabel('x (um)')
    #	plt.xlim(-10 , 10)
    N2 = floor(size(Mir_x)/2)
    plot(Det_s * 1e6, Amp(E1List[0,:])**2)
    #plot(Amp(E1List[0,:])**2)

    plt.show()

