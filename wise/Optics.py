# -*- coding: utf-8 -*-
"""
Created on Mon Aug 08 16:10:57 2016

@author: Mic
"""
import numpy as np
from numpy import *
from numpy.linalg import norm
import wise.Noise2 as Noise
from wise.Rayman5 import Range, FastResample1d, RotXY, _MatchArrayLengths

#==============================================================================
# 	CLASS: GaussianSource_1d
#==============================================================================
class GaussianSource_1d(object):
	#================================================
	# 	__init__
	#================================================
	def __init__(self, Lambda, Waist0, ZOrigin = 0, YOrigin = 0, Theta = 0 ):
		self.Lambda = Lambda
		self.Waist0 = Waist0

		self.ZOrigin = ZOrigin
		self.YOrigin = YOrigin
		self.RhoZOrigin = array([YOrigin, ZOrigin])
		self.ThetaPropagation = Theta

	def Fwhm(self,z):
		return self.Waist(z) * 2 * sqrt(np.log(2))

	#================================================
	# 	RayleighRange
	#================================================
	@property
	def RayleighRange(self):
		return   np.pi * self.Waist0**2 / self.Lambda
	#================================================
	# 	ThetaDiv
	#================================================
	@property
	def ThetaDiv(self):
		return  self.Lambda/np.pi/self.Waist0

	#================================================
	# 	WaistZ
	#================================================
	def Waist(self, z):
		return self.Waist0 * sqrt(1+ (z/self.RayleighRange)**2)

	#================================================
	# 	RCurvature
	#================================================
	def RCurvature(self, z):
		return z * (1+(self.RayleighRange/z)**2)

	#================================================
	# 	GouyPhase
	#================================================
	def GouyPhase(self,z):
		return np.arctan(z/self.RayleighRange)

	#================================================
	# 	Phase
	#================================================
	def Phase(self, z,y):
		k = 2 * pi / self.Lambda
		Ph = (k*z + k *y**2/2/self.RCurvature(z) - self.GouyPhase(z))
		ZeroPos = (z==0)  # gestisto eventuale singolarità nella fase
		Ph[ZeroPos] = 0
		return Ph
	#================================================
	# 	Cycles
	#================================================
	def Cycles(self,x = np.array(None) , z = np.array(None)):
		'''
		z: distance (m) scalar
		x: 1darray, sampling
		'''
		return np.cos(self.Phase(x,z))

	#================================================
	# 	Eval
	#================================================
	def Eval(self,z = np.array(None) , y = np.array(None)):
		'''
			z ed y sono nel sistema di riferimento della gaussiana
		'''
		(z,y) = _MatchArrayLengths(z,y)

		k = 2 * pi / self.Lambda

		Ph = (k*z + k *y**2/2/self.RCurvature(z) - self.GouyPhase(z))
#		Ph = (k*y**2/2/self.RCurvature(z) )
		ZeroPos = (z==0)  # gestisto eventuale singolarità nella fase
		Ph[ZeroPos] = 0
		Norm = 	(self.Waist0 / self.Waist(z))
		A = np.exp(-y**2/self.Waist(z)**2)
		return Norm * A *	np.exp(+1j*Ph)

	EvalField_XYOb = Eval


	#================================================
	# 	EvalField
	#================================================
	def EvalField_XYLab(self, x = np.array(None), y = np.array(None)):
		(x,y) = _MatchArrayLengths(x,y)
		'''
		in qusta funciton ho introdotto il concetto per cui
		x è la coordinata nel sistema di riferimento esterno
		z è la distanza dal waist.
		Questo fa si Waist(z), Radius(z) e simili vogliano l z

		'''

		#codice che dovrebbe funzionare ma che non va
		'''
		# cambio di coordinate da (x,y) lab a (xg,yg)
		XYOrigin = np.array([self.ZOrigin, self.YOrigin])
		[zg, yg] = CartChange(x,y, NewOrigin = XYOrigin, Theta = self.ThetaPropagation)
		'''
		# pezza
		myOrigin = array([self.ZOrigin, self.YOrigin])
		[zg,yg] = RotXY(x,y, Origin = myOrigin, Theta = - self.ThetaPropagation)
		zg = zg - myOrigin[0]
		yg = yg - myOrigin[1]

		return self.EvalField_XYOb(zg,yg)

# 	END CLASS: GaussianSource_1d
#==============================================================================

#==============================================================================
# 	CLASS: Ellipse
#==============================================================================
class Ellipse(object):
	'''
	equazione dell'ellisse:
	x^2/a^2 + y^2/b^2 = 1
	'''
	class _ClassOptions(object):
		def __init__(self):
			self.AUTO_UPDATE_MIRROR_LENGTH = True
			self.USE_FIGUREERROR = True
			self.USE_ROUGHNESS = False

	#================================
	# INIT
	#================================
	def __init__(self, a =None ,b = None, f1 = None, f2 = None, Alpha = None, L = None,
					MirXMid = None):
		self._FigureErrors = []
		self._FigureErrorSteps = []
		self._Roughness = Noise.RoughnessMaker()
		self.Options = Ellipse._ClassOptions()

		# Serie di parametri #1
		if all([arg != None for arg in [a,b, MirXMid, L]]):
			self._a = a
			self._b = b
			self._c = sqrt(a**2 - b**2)
			self.XYF1 = [-self.c, 0]
			self.XYF2 = [self.c,0]

			self._SetCoordinates(MirXMid, L)
			self._RefreshGeometricParameters()

			self._Alpha = abs(self.p1_Angle) + abs(self.pTan_Angle)

		# Serie di parametri 2
		elif all([arg != None for arg in [f1,f2,Alpha,L]]):

			self._f1 = f1
			self._f2 = f2
			self._Alpha = Alpha
			self._L = L

			self._a = 0.5*(f1+f2)
			self._c = 0.5 * sqrt(cos(Alpha)**2 * (f1+f2)**2 + sin(f1-f2)**2)
			self._c = 0.5 * sqrt(f1**2 + f2**2 - 2*f1*f2*cos(pi - 2*Alpha))
			self._b = sqrt(self.a**2 - self.c**2)
			elle = 2*self._c
			self.Theta = np.arcsin(self._f2/elle * sin(pi - 2*Alpha))
			self.XYF1 = [-self.c, 0]
			self.XYF2 = [self.c,0]

			XMid = f1*cos(self.Theta) + self.XYF1[0]
			YMid = self.EvalY(XMid)

			self.XYMid = array([XMid, YMid])
			self._SetMirrorCoordinates(self.XYMid[0], L)
			self._RefreshGeometricParameters()

	#================================
	# __disp__
	#================================
	def __str__(self):
		s = '\n a=%0.2f\n b=%0.2f\n c=%0.2f\n f1=%0.2f\n f2=%0.2f\n\n' %(self.a, self.b, self.c, self.f1, self.f2)
		return s



	#================================
	# _RefreshGeometricParameters
	#================================
	def _RefreshGeometricParameters(self):
		'''
			Trova equazioni dei due bracci
		'''
		# Trovo asse Sorgente- Centro Specchio (da mettere nella classe)
		[p1, p2] = self.TraceRay(self.XYF1, self.XYMid)
		self._p1 = p1
		self._p2 = p2
		self._p1_Angle = arctan(self.p1[0])
		self._p2_Angle = arctan(self.p2[0])


		# equazione della tangente al centro dello specchio

		m = - self.b**2 / self.a**2 * self.XYMid[0] / self.XYMid[1]
		q =  self.b**2 / self.XYMid[1]

		self._pTan = array([m,q])
		self._pTan_Angle = arctan(m)







	#================================
	# _SetMirrorCoordinates
	#================================
	def _SetMirrorCoordinates(self, XMid, L):
		'''
		Dati La posizione dello specchio e la lunghezza, definisce
		XYStart e XYEnd.
		Versione aggiornata di SetCoordinates (che si potrebbe cancellare)
		'''

		XStart = XMid - 0.5*L
		self.XYStart = array([XStart, self.EvalY(XStart)])
		XEnd = XMid + 0.5* L
		self.XYEnd  = array([XEnd, self.EvalY(XEnd)])
		self._L = L

	#================================
	# _SetCoordinates
	#================================
	def _SetCoordinates(self, XMid, DeltaX):
		'''
		Dati La posizione dello specchio e la lunghezza, definisce f1 e f2
		'''
		YMid = self.EvalY(XMid)
		XStart = XMid - 0.5*DeltaX
		YStart = self.EvalY(XStart)
		XEnd = XMid + 0.5*DeltaX
		YEnd= self.Eval(XEnd)

		self.XYMid = array([XMid, YMid])
		self.XYStart = array([XStart, YStart])
		self	.XYEnd = array([XEnd, YEnd])

		self._f1 = norm(self.XYMid - self.XYF1)
		#self._f1 = sqrt((self.XYMid[0] - self.XYF1[0])**2 + (self.XYMid[1] - self.XYF1[1])**2)
		self._f2 = norm(self.XYMid - self.XYF2)
		#self._f2 = sqrt((self.XYMid[0] - self.XYF2[0])**2 + (self.XYMid[1] - self.XYF2[1])**2)
		self._L = DeltaX

	#================================
	# SetFocalLenghts
	#================================
	def SetFocalLengths(self, f1, f2):
		'''
			For a given couple of focal lengths, it automatically sets the start, the
			midst and the end coordinates of the ellipse.
		'''
		self._f1 = f1
		self._f2 = f2
		CosGamma = (f1**2 + (2*self.c)**2 - f2**2) / 4/self.c/f1 ;
		xMid = f1 * CosGamma - self.c ;
		yMid = self.Eval(xMid)

		self.XYMid = [xMid, yMid]



	#================================
	# FigureErrorAdd
	#================================
	def FigureErrorAdd(self, h, Step = 1e-3):
		'''
		Aggiunge un array1d alla lista dei Figure error.
		h viene memorizzato in FigureErrors
		Step viene memorizzato in FigureErrorSteps
		'''
		self._FigureErrors.append(h)
		self._FigureErrorSteps.append(Step)

		if self.Options.AUTO_UPDATE_MIRROR_LENGTH:
			NewL = len(h)*Step
			self._SetMirrorCoordinates(self.XYMid[0], NewL)


	#================================
	# FigureErrorRemove
	#================================
	def FigureErrorRemove(self,i):
		self._FigureErrors.remove(i)






	#================================
	# EvalY (x)
	#================================
	def EvalY(self, x, Sign = +1):
		'''
		Valuta l'equazione dell'ellisse
		'''
		x = array(x)
		tmp = Sign*self.b * sqrt(1 - x**2 / self.a**2)
		return tmp

	Eval = EvalY

	#================================
	# GetXY_FocalPlaneAtF2(Size,N)
	#================================
	def GetXY_TransversePlaneAtF2(self, Length,N, Defocus = 0 ):
		'''
			Uses: XYF1, XYF2, XYMid
			Length (m)
			N: # samples
		'''
		Size = Length
		[p1, p2] = self.TraceRay(self.XYF1, self.XYMid)
		m = -1/p2[0]
		theta = arctan(m)
		thetaNorm = arctan(p2[0])
		DeltaXY = Defocus * array([cos(thetaNorm), sin(thetaNorm)])
		XY = self.XYF2 + DeltaXY
#		q = - self.XYF2[0] * m
		q = XY[1] - XY[0] * m
		p = array([m,q])

		Det_x0 = XY[0] - Size/2 * cos(theta)
		Det_x1 = XY[0] + Size/2 * cos(theta)
		x = np.linspace(Det_x0, Det_x1,N)
		y = polyval(p,x)
		return [x,y]



	#================================
	# GetXY_IdealMirror(N)
	#================================
	def GetXY_IdealMirror(self, N, Sign = +1):
		'''
			Evaluates the Ellipse only over the physical support of the mirror
			(within XStart and XEnd)
			N is the number of samples in X.

			 @TODO: define N of samples along the ellipse (ma serve?)
		'''
		x = np.linspace(self.XYStart[0], self.XYEnd[0], N)
		return [x,self.Eval(x, Sign)]






	#================================
	# GetXY_MeasuredMirror
	#================================
	def GetXY_MeasuredMirror(self,N, iFigureError = 0, GenerateRoughness = False ):

		# carico il figure error e, se necessario, lo ricampiono
		#-----------------------------------------------------------------
		if len(self._FigureErrors)-1 >= iFigureError:
			hFigErr  = self.FigureErrors[iFigureError]
			self._L = len(hFigErr) * self._FigureErrorSteps[iFigureError]
			hFigErr  = FastResample1d(hFigErr - np.mean(hFigErr  ), N)
		else:
			hFigErr   = np.zeros(N)


		# aggiungo la roughness (se richiesto, rigenero il noise pattern)
		#-----------------------------------------------------------------


		if self.Options.USE_ROUGHNESS == True:
			dx = self.L/N
			hRoughness = self.Roughness.Generate(N, dx)
			myResidual = hFigErr + hRoughness
		else:
			myResidual = hFigErr
		# proiezione del FigError sull'ellisse
		#-----------------------------------------------------------------
		Mir_x, Mir_y = self.GetXY_IdealMirror(N)
		ThetaList = self._LocalTangent(Mir_x, Mir_y)
		Mir_xx = Mir_x + hFigErr * sin(ThetaList)
		Mir_yy = Mir_y + hFigErr * cos(ThetaList)


		return Mir_xx, Mir_yy

	#================================
	# _AddResidualToEllipse
	#================================
	def _AddResidualToEllipse(self, myResidual):
		# Assume che la lunghezza fisica di myResidual sia uguale a quella di self.L (che è ciò che accate se Options.)
		N = size(myResidual)
		[Mir_x, Mir_y] = self.GetXY_IdealMirror(N)
		ThetaList = self._LocalTangent(Mir_x, Mir_y)
		NewMir_x = Mir_x + myResidual * sin(ThetaList)
		NewMir_y = Mir_y + myResidual * cos(ThetaList)
		return (NewMir_x, NewMir_y)




	#================================
	# PROP:
	#================================
	@property
	def Roughness(self):
		return self._Roughness

	# se mantengo le cose così, dovrei aggiungere anche XYMid, fuochi etc... riflettere

	@property
	def xy(self): return
	@property
	def FigureErrors(self): return self._FigureErrors
	@property
	def FigureErrorSteps(self):
		'''

		'''
		return self._FigureErrorSteps

	@property
	def a(self):		return self._a

	@property
	def b(self):		return self._b

	@property
	def c(self):		return self._c

	@property
	def f1(self):		return self._f1

	@property
	def f2(self):		return self._f2

	@property
	def Alpha(self):		return self._Alpha

	@property
	def L(self):		return self._L

	@L.setter
	def L(self, val): self._L = val

	@property # retta braccio 1
	def p1(self):		return self._p1

	@property # retta braccio 2
	def p2(self): 	return self._p2

	@property
	def p1_Angle(self): return self._p1_Angle

	@property
	def p2_Angle(self): return self._p2_Angle

	@property # retta tangente al centro specchio
	def pTan(self): 	return self._pTan

	@property
	def pTan_Angle(self): 	return self._pTan_Angle


	#================================
	# TraceRay
	#================================
	def TraceRay(self, Start, End):
		''' Funzione scritta un po' male perché porting da MATLAB, pensato per
			fare cose leggermente diverse.

			Dato l'ellisse (oggetto), il punto di partenza PStart e la X di incidenza,
			trova il polinomio di ordine 1 (retta) del raggio incidente e di quello
			riflesso
		'''

		a = self.a
		b = self.b
		if size(End) == 1:
			xEll = End
			yEll = self.EvalY(xEll)
		elif size(End) == 2:
			xEll = End[0]
			yEll = End[1]



		xStart = Start[0] ;
		yStart = Start[1] ;

		# raggio uscente (2)


		m0 = -b**2/a**2 * xEll/yEll ;
		m2 =  (yStart - yEll) / (xStart - xEll) ;

		Theta0 = arctan(m0) ;
		Theta2 = arctan2((yStart - yEll),(xStart - xEll)) ;
		Theta1 =+( -Theta2 + 2*Theta0 - pi) ;

		m1 = tan(Theta1) ;


		q = yEll - m1 * xEll ;

		p = [m1,q] ;
		p1 = p ;			# il raggio (1) (entrante)

		xList2 = Range(-2*a,xEll,0.001) ;
		yList2 = polyval(p, xList2) ;


		# raggio di partenza (1) ;

		q = yEll - m2 * xEll ;

		p = [m2,q] ;
		p2 = p ; 		# il raggio (2) uscente
		xList1 = Range(xEll, 2*a, 0.001)
		yList1 = polyval(p, xList1) ;

		xList = [xList1, xList2] ;
		yList = [yList1, yList2] ;

		return p2, p1
	#================================
	# _Alpha_to_Theta
	#================================
	def _Alpha_to_Theta(self,Alpha):
		m = -self.b**2/self.a**2 * self.XYMid[0] /self.XYMid[1]
		return abs(Alpha - abs(m ))

	#================================
	# _Theta_to_Alpha
	#================================
	def _Theta_to_Alpha(self,Theta):
		m = -self.b**2/self.a**2 * self.XYMid[0] /self.XYMid[1]
		return abs(Theta - abs(arctan(m)))

	#================================
	# _LocalTangent
	#================================
	def _LocalTangent(self, x0, y0):
		m = -self.b**2 / self.a**2 * x0/y0
		return arctan(m)

		#=======================================================================