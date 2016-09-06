# -*- coding: utf-8 -*-
"""
Created on Thu Jul 07 14:08:31 2016

@author: Mic
"""

from numpy import *
import numpy as np
import wise.Rayman5 as rm
Gauss1d =  lambda x ,y : None
from scipy import interpolate as interpolate

class PsdFuns:
	@staticmethod
	def Flat(x, *args):
		N = len(x)
		return np.zeros([1,N]) +1
	@staticmethod
	def PowerLaw(x,a,b):
		return a*x**b
	@staticmethod
	def Gaussian(x,sigma, x0=0):
		return np.exp(-0.5 * (x-x0)**2/sigma**2)
	@staticmethod
	def Interp(x, xData, yData):
		f = interpolate.interp1d(xData, yData)

		return f(x)


def PsdFun2Noise_1d(N,dx, PsdFun, PsdArgs):
	'''
		Generates a noise pattern based an the Power spectral density returned
		by PsdFun
	'''
	x = np.arange(0,N//2+1, dx)
	yHalf = PsdFun(x, *PsdArgs)
	y = Psd2Noise_1d(yHalf, Semiaxis = True 	)
	return  x,y



#============================================================================
#	FUN: 	Psd2Noise
#============================================================================
def Psd2Noise_1d(Psd, Semiaxis = True, Real = True):
	'''
	Generates a noise pattern whose Power Spectral density is given by Psd.

	Parameters:
		Psd: 1darr
		Semiaxis:
			0: does nothing
			1: halven Pds, then replicate the halven part for left frequencies,
				producing an output as long as
			2: replicates all Pds for lef frequencies as well, producing an output
				twice as long as Psd
		Real: if True, the real part of the output is returned
	Returns:
		arr of the same length of Psd
	'''

	if Semiaxis == True:
		yHalf = Psd
		Psd = np.hstack((yHalf[-1:0:-1], 0, yHalf[1:-1]))
	y = np.fft.fftshift(Psd)
	r = 2*pi * np.random.rand(len(Psd))
	f = np.fft.ifft(y * exp(1j*r))

	if Real:
		return real(f)
	else:
		return f

#============================================================================
#	FUN: 	NoNoise_1d
#============================================================================
def NoNoise_1d(N, *args):
	return np.zeros([1,N])

#============================================================================
#	FUN: 	GaussianNoise_1d
#============================================================================
def GaussianNoise_1d(N,dx, Sigma):
	'''
	PSD(f) = exp(-0.5^f/Sigma^2)
	'''
	x = np.linspace( - N//2 *dx, N//2-1 * dx,N)
	y = exp(-0.5*x**2/Sigma**2)
	return Psd2Noise_1d(y)


#============================================================================
#	FUN: 	PowerLawNoise_1d
#============================================================================
def PowerLawNoise_1d(N, dx, a, b):
	'''
	PSD(x) = a*x^b
	'''
	x = np.arange(0,N//2+1, dx)
	yHalf = a * x**b
	y = np.hstack((yHalf[-1:0:-1], 0, yHalf[1:-1]))
	return Psd2Noise_1d(y, Semiaxis = True)

#============================================================================
#	FUN: 	CustomNoise_1d
#============================================================================
def CustomNoise_1d(N, dx, xPsd, yPsd):
	xPsd_, yPsd_ = rm.FastResample1d(xPsd, yPsd,N)
	return Psd2Noise_1d(yPsd_, Semiaxis = True)

#============================================================================
#	CLASS: 	NoiseGenerator
#============================================================================
class PsdGenerator:
	NoNoise = staticmethod(NoNoise_1d)
	Gauss  = staticmethod(GaussianNoise_1d)
	PowerLaw = staticmethod(PowerLawNoise_1d)
	NumericArray = staticmethod(CustomNoise_1d)



#============================================================================
#	FUN: 	FitPowerLaw
#============================================================================
def FitPowerLaw(x,y):
	import scipy.optimize as optimize

	fFit = lambda p, x: p[0] * x ** p[1]
	fErr = lambda p, x, y: (y - fFit(p, x))

	p0 = [11.0, 1.0]
	out = optimize.leastsq(fErr, p0, args=(x, y), full_output=1)

	pOut = out[0]

	b = pOut[1]
	a = pOut[0]

#	indexErr = np.sqrt( covar[0][0] )
#	ampErr = np.sqrt( covar[1][1] ) * amp

	return a,b

#==============================================================================
# 	CLASS: Roughness
#==============================================================================

class RoughnessMaker(object):

	class Options():
		FIT_NUMERIC_DATA_WITH_POWER_LAW  = True
		AUTO_ZERO_MEAN_FOR_NUMERIC_DATA = True
		AUTO_FILL_NUMERIC_DATA_WITH_ZERO  = True
		AUTO_RESET_CUTOFF_ON_PSDTYPE_CHANGE = True

	def __init__(self):
		self.PsdType = PsdFuns.PowerLaw
		self.PsdParams = np.array([1,1])
		self._IsNumericPsdInFreq = None
		self.CutoffLowHigh = [None, None]
		return None

	@property
	def PsdType(self):
		return self._PsdType
	@PsdType.setter
	def PsdType(self, Val):
		'''
		Note: each time that the Property value is set, self.CutoffLowHigh is
		reset, is specified by options
		'''
		self. _PsdType = Val
		if self.Options.AUTO_RESET_CUTOFF_ON_PSDTYPE_CHANGE == True:
			self.PsdCutoffLowHigh  = [None, None]

	#======================================================================
	# 	FUN: PdfEval
	#======================================================================
	def PsdEval(self, N, df, CutoffLowHigh = [None, None]):
		'''
		Evals the PSD in the range [0 - N*dx]
		It's good custom to have PSD[0] = 0, so that the noise pattern is
		zero-mean.
		Parameters:
			N: #of samples
			df: spacing of spatial frequencies
			LowCutoff: if >0, then Psd(f<Cutoff) is set to 0.
						if None, then LowCutoff = min()
		Returns:
			x : arr (spatial frequencies)
			yPsd:arr (Psd)
		'''

		'''
		The Pdf is evaluated only within LowCutoff and HoghCutoff
		If the Pdf is PsdFuns.Interp, then LowCutoff and HighCutoff are
		automatically set to min and max values of the experimental data
		'''
		def GetInRange(fAll, LowCutoff, HighCutoff):
			_tmpa  = fAll >= LowCutoff
			_tmpb = fAll <= HighCutoff
			fMid_Pos  = np.all([_tmpa, _tmpb],0)
			fMid = fAll[fMid_Pos]
			return fMid_Pos, fMid

		LowCutoff, HighCutoff = CutoffLowHigh
		fAll = np.linspace(0, N*df, N)
		yPsdAll = fAll* 0

		LowCutoff = 0 if LowCutoff == None else LowCutoff
		HighCutoff = N*df if HighCutoff == None else HighCutoff



		# Numeric PSD
		# Note: by default returned yPsd is always 0 outside the input data range
		if self.PsdType == PsdFuns.Interp:
			# Use Auto-Fit + PowerLaw
			if self.Options.FIT_NUMERIC_DATA_WITH_POWER_LAW == True:
					xFreq,y = self.NumericPsdGetXY()
					p = FitPowerLaw(1/xFreq,y)
					_PsdParams = p[0], -p[1]
					LowCutoff =  np.amin(self._PsdNumericX)
					HighCutoff = np.amin(self._PsdNumericX)
					fMid_Pos, fMid = GetInRange(fAll, LowCutoff, HighCutoff)
					yPsd = PsdFuns.PowerLaw(fMid, *_PsdParams )
			# Use Interpolation
			else:
				# check Cutoff
				LowVal =  np.amin(self._PsdNumericX)
				HighVal = np.amin(self._PsdNumericX)
				LowCutoff = LowVal if LowCutoff <= LowVal else LowCutoff
				HighCutoff = HighVal if HighCutoff >= HighVal else HighCutoff
				fMid_Pos, fMid = GetInRange(fAll, LowCutoff, HighCutoff)
				yPsd = self.PsdType(fMid, *self.PsdParams)

		# Analytical Psd
		else:
			fMid_Pos, fMid = GetInRange(fAll, LowCutoff, HighCutoff)
			yPsd = self.PsdType(fMid, *self.PsdParams)

		# copying array subset
		yPsdAll[fMid_Pos] = yPsd

		return fAll, yPsdAll

	#======================================================================
	# 	FUN: _FitNumericPsdWithPowerLaw
	#======================================================================
# in disusos
	def _FitNumericPsdWithPowerLaw(self):
		x,y = self.NumericPsdGetXY()
		if self._IsNumericPsdInFreq == True:
			p = FitPowerLaw(1/x,y)
			self.PsdParams = p[0], -p[1]
		else:
			p = FitPowerLaw(x,y)
			self.PsdParams = p[0], p[1]

	#======================================================================
	# 	FUN: MakeProfile
	#======================================================================
	def MakeProfile(self,N ,df):
		'''
			Evaluates the psd according to .PsdType, .PsdParams and .Options directives
			Returns an evenly-spaced array.
			If PsdType = NumericArray, linear interpolation is performed.

			parameters:
				N: # of samples
				dx: spacing
			returns:
				1d arr
		'''

#		x = np.linspace(0, (N//2 +1) *dx,(N//2 +1))
		f, yPsd = self.PdfEval(N,df)

		# Special case
#		if self.Options.FIT_NUMERIC_DATA_WITH_POWER_LAW == True:
#			self.PsdParams = list(FitPowerLaw(*self.NumericPsdGetXY()))
#			yPsd = PsdFuns.PowerLaw(x, *self.PsdParams)
#		else: # general calse
#			yPsd = self.PsdType(x, *self.PsdParams)

		yRoughness  = Psd2Noise_1d(yPsd, Semiaxis = True)
		return yRoughness

#		x = np.linspace(0, N*dx,N)
#		# Special case
#		if self.Options.FIT_NUMERIC_DATA_WITH_POWER_LAW == True:
#			self.PsdParams = list(FitPowerLaw(*self.NumericPsdGetXY()))
#			y = PowerLawNoise_1d(N, dx, *self.PsdParams)
#		else: # general calse
#			y = self.PsdType(N,dx, *self.PsdParams)
#		return y

	def NumericPsdSetXY(self,x,y):
		self._PsdNumericX = x
		if self.Options.AUTO_ZERO_MEAN_FOR_NUMERIC_DATA == True:
			y = y - np.mean(y)
		self._PsdNumericY = y

	def NumericPsdGetXY(self):
		return self._PsdNumericX, self._PsdNumericY
	def NumericPsdLoadXY(self, FilePath, xScaling = 1, yScaling = 1 , xIsSpatialFreq = False):
		''' @TODO: specificare formati e tipi di file
		'''
		self._IsNumericPsdInFreq = xIsSpatialFreq
		s = np.loadtxt(FilePath)
		x = s[:,0]
		y = s[:,1]
		# array sorting
		i = np.argsort(x)
		x = x[i]
		y = y[i]

		x = x * xScaling
		y = y * yScaling
		# inversion of x-axis if not spatial frequencies
		if xIsSpatialFreq == False:
			x = 1/x

		self.PsdCutoffLowHigh = [np.amin, np.amax(x)]
		self.PsdType = PsdFuns.Interp
		self.PsdParams = [x,y]


		# Auto-set
		# fill 0-value (DC Component)
#		if self.Options.AUTO_FILL_NUMERIC_DATA_WITH_ZERO == True:
#			if np.amin(x >0):
#				x = np.insert(x,0,0)
#				y = np.insert(y,0,0)	# 0 in psd => 0-mean value in the noise pattern


		# sync other class values
		self.NumericPsdSetXY(x, y)

	def Generate(self, N = None, dx = None, CutoffLowHigh = [None, None]):
		'''
		Parameters
			N: # of output samples
			dx: step of the x axis
		Note: generates an evenly spaced array
		'''
		L = dx * N
		df = 1/L
		fPsd, yPsd = self.PsdEval(N//2 +1  , df = df,
										CutoffLowHigh = CutoffLowHigh )
		h = Psd2Noise_1d(yPsd, Semiaxis = True)

		return  h

