# -*- coding: utf-8 -*-

"""
Module for writing spectrometer output to a SDFITS file.  The SDFITS created by this 
modulefiles closely follow the Parkes variant of the SDFITS convention 
(see [http://fits.gsfc.nasa.gov/registry/sdfits.html]).  The main differences between 
the two is that the LWA SDFITS do not contain calbration or weather information.  This,
however, should not stop the files from being loaded into CASA with the ATNF Spectral 
Analysis Package (ASAP).

.. versionchanged:: 0.5.0
	The classes and functions defined in this module are based heavily off 
	the :mod:`lsl.writer.fitsidi` writer.
"""

import os
import gc
import re
import numpy
import pyfits
import warnings
from datetime import datetime

import astro
#from lsl.common.stations import lwa1

__version__ = '0.5'
__revision__ = '$Rev$'
__all__ = ['SD', '__version__', '__revision__', '__all__']


SDVersion = (1, 6)

StokesCodes = { 'I':  1,  'Q': 2,   'U':  3,  'V':  4, 
			'RR': -1, 'LL': -2, 'RL': -3, 'LR': -4, 
			'XX': -5, 'YY': -6, 'XY': -7, 'YX': -8}

NumericStokes = { 1: 'I',   2: 'Q',   3: 'U',   4: 'V', 
			  -1: 'RR', -2: 'LL', -3: 'RL', -4: 'RL', 
			  -5: 'XX', -6: 'YY', -7: 'XY', -8: 'YX'}


class SD(object):
	"""
	Class for storing spectrometer data and writing the data, along with array
	frequency setup, etc., to a SDFITS file that can be read into CASA via the
	sd.scantable() function.
	"""
	
	class _Frequency:
		"""
		Holds information about the frequency setup used in the file.
		"""

		def __init__(self, offset, channelWidth, bandwidth):
			self.id = 1
			self.bandFreq = offset
			self.chWidth = channelWidth
			self.totalBW = bandwidth
			self.sideBand = 1
			self.baseBand = 0
	
	class _SpectrometerData(object):
		"""
		Representns one spectrum for a given observation time.
		"""
		
		def __init__(self, obsTime, intTime, dataDict, pol=StokesCodes['XX'],
                             rfiDict=None,
                             rfiMaskDict=None,
                             flagDict=None):
                        
			self.obsTime = obsTime
			self.intTime = intTime
			self.dataDict = dataDict
                        if rfiDict:
                                self.rfiDict=rfiDict
                        if rfiMaskDict:
                                self.rfiMaskDict=rfiMaskDict
                        if flagDict:
                                self.flagDict=flagDict                                
			self.pol = pol
		
		def time(self):
			return self.obsTime
	
	def parseRefTime(self, refTime):
		"""
		Given a time as either a integer, float, string, or datetime object, 
		convert it to a string in the formation 'YYYY-MM-DDTHH:MM:SS'.
		"""

		# Valid time string (modulo the 'T')
		timeRE = re.compile(r'\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?')

		if type(refTime) in (int, long, float):
			refDateTime = datetime.utcfromtimestamp(refTime)
			refTime = refDateTime.strftime("%Y-%m-%dT%H:%M:%S")
		elif type(refTime) == datetime:
			refTime = refTime.strftime("%Y-%m-%dT%H:%M:%S")
		elif type(refTime) == str:
			# Make sure that the string times are of the correct format
			if re.match(timeRE, refTime) is None:
				raise RuntimeError("Malformed date/time provided: %s" % refTime)
			else:
				refTime = refTime.replace(' ', 'T', 1)
		else:
			raise RuntimeError("Unknown time format provided.")

		return refTime

	def refTime2AstroDate(self):
		"""
		Convert a reference time string to an :class:`lsl.astro.date` object.
		"""

		dateStr = self.refTime.replace('T', '-').replace(':', '-').split('-')
		return astro.date(int(dateStr[0]), int(dateStr[1]), int(dateStr[2]), int(dateStr[3]), int(dateStr[4]), float(dateStr[5]))

	def __init__(self, filename, refTime=0.0, verbose=False, memmap=None, clobber=False):
		"""
		Initialize a new SDFITS object using a filename and a reference time 
		given in seconds since the UNIX 1970 ephem, a python datetime object, or a 
		string in the format of 'YYYY-MM-DDTHH:MM:SS'.
		
		.. versionchanged:: 1.1.2
			Added the 'memmap' and 'clobber' keywords to control if the file
			is memory mapped and whether or not to overwrite an existing file, 
			respectively.
		"""

		# File-specific information
		self.filename = filename
		self.verbose = verbose

		# Observation-specific information
		self.site = '' #lwa1
		self.refTime = self.parseRefTime(refTime)
		self.nChan = 0
		self.nStokes = 0
		self.refVal = 0
		self.refPix = 0
		self.channelWidth = 0

		# Parameters that store the meta-data and data
		self.freq = []
		self.stokes = []
		self.data = []

		# Misc.
		self.tSys = 250
		self.observer = 'UKNOWN'
		self.project = 'UNKNOWN'
		self.mode = 'UNKNOWN'
		

		# Open the file and get going
		if os.path.exists(filename):
			if clobber:
				os.unlink(filename)
			else:
				raise IOError("File '%s' already exists" % filename)
		self.FITS = pyfits.open(filename, mode='append', memmap=memmap)

	def setSite(self, site):
		"""
		Set the TELESCOP keyword in the primary HDU using an :class:`lsl.common.stations.LWAStation`
		object.
		"""

		self.site = site
		
	def setStokes(self, polList):
		"""
		Given a list of Stokes parameters, update the object's parameters.
		"""

		for pol in polList:
			if isinstance(pol, str) or isinstance(pol, unicode):
				numericPol = StokesCodes[pol.upper()]
			else:
				numericPol = pol
				
			if numericPol not in self.stokes:
				self.stokes.append(numericPol)

		# Sort into order of 'XX', 'YY', 'XY', and 'YX' or 'I', 'Q', 'U', and 'V'
		self.stokes.sort()
		if self.stokes[0] < 0:
			self.stokes.reverse()

		self.nStokes = len(self.stokes)

	def setFrequency(self, freq):
		"""
		Given a numpy array of frequencies, set the relevant common observation
		parameters and add an entry to the self.freq list.
		"""

		self.nChan = len(freq)
		self.refVal = freq[0]
		self.refPix = 1
		self.channelWidth = numpy.abs(freq[1] - freq[0])
		totalWidth = numpy.abs(freq[-1] - freq[0])

		freqSetup = self._Frequency(0.0, self.channelWidth, totalWidth)
		self.freq.append(freqSetup)
		
	def setObserver(self, observer, project='UNKNOWN', mode='UNKNOWN'):
		"""
		Set the observer name, project, and observation mode (if given) to the 
		self.observer, self.project, and self.mode attributes, respectively.
		"""
		
		self.observer = observer
		self.project = project
		self.mode = mode

	def addDataSet(self, obsTime, intTime, beam, data,
                       pol='XX',
                       rfi=None,
                       rfi_mask=None,
                       flag=None):
		"""
		Create a SpectrometerData object to store a collection of spectra.
		"""
		
		if type(pol) == str:
			numericPol = StokesCodes[pol.upper()]
		else:
			numericPol = pol
		
		dataDict = {}
                rfiDict = {}
                rfiMaskDict = {}
                flagDict = {}
                
		for i,b in enumerate(beam):
			dataDict[b] = data[i,:]                
                        if not rfi is None:
                                rfiDict[b] = rfi[i,:]
                        if not rfi_mask is None:
                                rfiMaskDict[b] = rfi_mask[i,:]
                        if not flag is None:
                                flagDict[b] = flag[i,:]
                        
		self.data.append( self._SpectrometerData(obsTime, intTime, dataDict,
                                                         pol=numericPol,
                                                         rfiDict=rfiDict,
                                                         rfiMaskDict=rfiMaskDict,
                                                         flagDict=flagDict) )

	def write(self):
		"""
		Fill in the SDFITS file will all of the tables in the correct order.
		"""
		
		def __sortData(x, y):
			"""
			Function to sort the self.data list in order of time and then 
			polarization code.
			"""
			
			xID = x.obsTime*10000000 + abs(x.pol)
			yID = y.obsTime*10000000 + abs(y.pol)
			
			if xID > yID:
				return 1
			elif xID < yID:
				return -1
			else:
				return 0
				
		# Sort the data set
		self.data.sort(cmp=__sortData)
		
		self.__writePrimary()
		self.__writeData()
		
		# Clear out the data section
		del(self.data[:])
		gc.collect()

	def close(self):
		"""
		Close out the file.
		"""

		self.FITS.flush()
		self.FITS.close()

	def __writePrimary(self):
		"""
		Write the primary HDU to file.
		"""

		primary = pyfits.PrimaryHDU()
		
		primary.header['NAXIS'] = (0, 'indicates SD file')
		primary.header['EXTEND'] = (True, 'indicates SD file')
		ts = str(astro.get_date_from_sys())
		primary.header['DATE'] = (ts.split()[0], 'IDI file creation date')
		primary.header['ORIGIN'] = 'LSL SDFITS writer'
		primary.header['TELESCOP'] = (self.site.name, 'Telescope name')
		
		self.FITS.append(primary)
		self.FITS.flush()
		
	def __writeData(self):
		"""
		Define the SINGLE DISH table.
		"""
		
		scanList = []
		dateList = []
		timeList = []
		intTimeList = []
		beamList = []
		mList = []
                rfiList = []
                rfiMaskList = []
                flagList = []
		rawList = []
		scanCount = 1
                add_rfi=False
                add_rfi_mask=False
                add_flag=False
                
		for i,dataSet in enumerate(self.data):
			if dataSet.pol == self.stokes[0]:
				tempMList = {}
				for stokes in self.stokes:
					tempMList[stokes] = {}
                                #RFI
                                if hasattr(dataSet, 'rfiDict'):                         
                                        add_rfi=True
                                        tempRFIList = {}
                                        for stokes in self.stokes:                                        
                                                tempRFIList[stokes] = {}
                                                
                                #RFI_MASK
                                if hasattr(dataSet, 'rfiMaskDict'):
                                        add_rfi_mask=True
                                        tempRFIMaskList = {}
                                        for stokes in self.stokes:                                        
                                                tempRFIMaskList[stokes] = {}
                                #FLAG_MASK
                                if hasattr(dataSet, 'flagDict'):
                                        add_flag=True
                                        tempFlagList = {}
                                        for stokes in self.stokes:                                        
                                                tempFlagList[stokes] = {}
                                                
			beams = list(dataSet.dataDict.keys())
			beams.sort()
			for b in beams:
				specData = dataSet.dataDict[b]
				
				# Load the data into a matrix
				tempMList[dataSet.pol][b] = specData.ravel()
                                if add_rfi:
                                        tempRFIList[dataSet.pol][b] = dataSet.rfiDict[b].ravel()
                                if add_rfi_mask:
                                        tempRFIMaskList[dataSet.pol][b] = dataSet.rfiMaskDict[b].ravel()
                                if add_flag:
                                        tempFlagList[dataSet.pol][b] = dataSet.flagDict[b].ravel()
                                
				#if i==0:
                                #        print('sdfits: b,dataSet.pol,len(dataSet[b])',b,dataSet.pol,specData.ravel().shape)
                                        
				if dataSet.pol == self.stokes[0]:
					# Observation date and time
					#utc = astro.taimjd_to_utcjd(dataSet.obsTime)
                                        utc = dataSet.obsTime
					date = astro.get_date(utc)
					date.hours = 0
					date.minutes = 0
					date.seconds = 0
					utc0 = date.to_jd()
						
					scanList.append(scanCount)
					dateList.append('%4i-%02i-%02i' % (date.years, date.months, date.days))
					timeList.append((utc - utc0)*24*3600)
					intTimeList.append(dataSet.intTime)
					beamList.append(b)
					rawList.append(b)
			
			if dataSet.pol == self.stokes[-1]:
				for b in rawList:
					matrix = numpy.zeros((self.nStokes,self.nChan), dtype=numpy.float32)
					matrix_rfi = numpy.zeros((self.nStokes,self.nChan), dtype=numpy.float32)
					matrix_rfi_mask = numpy.zeros((self.nStokes,self.nChan), dtype=numpy.int)
					matrix_flag = numpy.zeros((self.nStokes,self.nChan), dtype=numpy.int) 
					for p in xrange(self.nStokes):
						try:
							matrix[p,:] = tempMList[self.stokes[p]][b]
                                                        if add_rfi:
                                                                matrix_rfi[p,:] = tempRFIList[self.stokes[p]][b]
                                                        if add_rfi_mask:
                                                                matrix_rfi_mask[p,:] = tempRFIMaskList[self.stokes[p]][b]
                                                        if add_flag:
                                                                matrix_flag[p,:] = tempFlagList[self.stokes[p]][b]
                                                                
						except KeyError:
							warnings.warn('Key mis-match %s %s' % (str(b),
                                                                                               str(tempMList[self.stokes[p]].keys())),
                                                                      RuntimeWarning)
							
					mList.append(matrix.ravel())
                                        if add_rfi:                                        
                                                rfiList.append(matrix_rfi.ravel())
                                        if add_rfi_mask:
                                                rfiMaskList.append(matrix_rfi_mask.ravel())
                                        if add_flag:
                                                flagList.append(matrix_flag.ravel()) 
				scanCount += 1
				rawList = []                
                
		# Scan number
		c1  = pyfits.Column(name='SCAN', format='1I', 
						array=numpy.array(scanList))
		## Cycle
		#c2 = pyfits.Column(name='CYCLE', format='1J', 
						#array=numpy.array([1 for s in scanList]))
		# DATE-OBS
		c3  = pyfits.Column(name='DATE-OBS', format='10A', 
						array = numpy.array(dateList))
		# Time elapsed since 0h
		c4  = pyfits.Column(name='TIME', format='1D', unit = 's', 
						array = numpy.array(timeList))
		# Integration time (seconds)
		c5  = pyfits.Column(name='EXPOSURE', format='1E', unit='s', 
						array=numpy.array(intTimeList, dtype=numpy.float32))
		# Object name
		c6  = pyfits.Column(name='OBJECT', format='16A', 
						array=numpy.array(['BELEIN_OBS' for s in scanList]))
		# Object position (deg and deg)
		c7  = pyfits.Column(name='OBJ-RA', format='1D', unit='deg', 
						array=numpy.array([0.0 for s in scanList]))
		c8  = pyfits.Column(name='OBJ-DEC', format='1D', unit='deg', 
						array=numpy.array([0.0 for s in scanList]))
		# Rest frequency (Hz)
		c9  = pyfits.Column(name='RESTFRQ', format='1D', unit='Hz', 
						array=numpy.array([0.0 for s in scanList]))
		# Observation mode
		c10 = pyfits.Column(name='OBSMODE', format='16A', 
						array=numpy.array([self.mode for s in scanList]))
		# Beam (tuning)
		c11 = pyfits.Column(name='BEAM', format='1I', 
						array=numpy.array(beamList))
		# IF
		c12 = pyfits.Column(name='IF', format='1I', 
						array=numpy.array([self.freq[0].id for s in scanList]))
		# Frequency resolution (Hz)
		c13 = pyfits.Column(name='FREQRES', format='1D', unit='Hz', 
						array=numpy.array([self.freq[0].chWidth for s in scanList]))
		# Bandwidth of the system (Hz)
		c14 = pyfits.Column(name='BANDWID', format='1D', unit='Hz', 
						array=numpy.array([self.freq[0].totalBW for s in scanList]))
		# Frequency axis - 1
		c15 = pyfits.Column(name='CRPIX1', format='1E',
						array=numpy.array([self.refPix for s in scanList]))
		c16 = pyfits.Column(name='CRVAL1', format='1D', unit='Hz', 
						array=numpy.array([self.refVal for s in scanList]))
		c17 = pyfits.Column(name='CDELT1', format='1D', unit='Hz', 
						array=numpy.array([self.freq[0].chWidth for s in scanList]))
		c18 = pyfits.Column(name='CRVAL3', format='1D', unit='deg', 
						array=numpy.array([0.0 for s in scanList]))
		# Dec. axis - 4
		c19 = pyfits.Column(name='CRVAL4', format='1D', unit='deg', 
						array=numpy.array([0.0 for s in scanList]))
		## Scan rate
		#c20 = pyfits.Column(name='SCANRATE', format='2E', unit='deg/s', 
						#array=numpy.array([[0,0] for s in scanList]))
						
		#
		# Calibration information (currently not implemented)
		#
		## System temperature  *** UNKNOWN ***
		#c21 =  pyfits.Column(name='TSYS', format='2E', unit='K', 
						#array=numpy.array([[self.tSys,self.tSys] for s in scanList]))
		## CALFCTR *** UNKNOWN ***
		#c22 =  pyfits.Column(name='CALFCTR', format='2E', unit='K', 
						#array=numpy.array([[1,1] for s in scanList]))
		
		# Data
		c23 = pyfits.Column(name='DATA', format='%iE' % (self.nStokes*self.nChan), unit='UNCALIB', 
						array=numpy.array(mList))

		#
		# RFI Simulation information
		#
                
		# RFI table
                if add_rfi:
                        c24 = pyfits.Column(name='RFI', format='%iE' % (self.nStokes*self.nChan), unit='UNCALIB', 
						array=numpy.array(rfiList))

		# RFI MASK table                        
                if add_rfi_mask:
                        c25 = pyfits.Column(name='RFI_MASK', format='%iB' % (self.nStokes*self.nChan), 
						array=numpy.array(rfiMaskList))		
                        
		# Flag table                        
                if add_flag:
                        c26 = pyfits.Column(name='FLAGGED', format='%iB' % (self.nStokes*self.nChan), 
						array=numpy.array(flagList))		
                else:
                        c26 = pyfits.Column(name='FLAGGED', format='%iB' % (self.nStokes*self.nChan), 
						array=numpy.array([[0,]*self.nStokes*self.nChan for s in scanList]))		
                        
						
		#
		# Data masking table (currently not implemented)
		#
		#
		# Calibration information (currently not implemented)
		#
		## TCAL *** UNKNOWN ***
		#c25 = pyfits.Column(name='TCAL', format='2E', unit='Jy', 
						#array=numpy.array([[1,1] for s in scanList]))
		## TCALTIME *** UNKNOWN ***
		#c26 = pyfits.Column(name='TCALTIME', format='16A', 
						#array=numpy.array(['UNKNOWN' for s in scanList]))
		
		#
		# Pointing information (currently not implemented)
		#
		## Azimuth *** UNKNOWN ***
		#c27 = pyfits.Column(name='AZIMUTH', format='1E', unit='deg', 
						#array=numpy.array([0 for s in scanList]))
		## Elevation *** UNKNOWN ***
		#c28 = pyfits.Column(name='ELEVATIO', format='1E', unit='deg', 
						#array=numpy.array([0 for s in scanList]))
		## Parallactic angle *** UNKNOWN ***
		#c29 = pyfits.Column(name='PARANGLE', format='1E', unit='deg', 
						#array=numpy.array([0 for s in scanList]))
		
		#
		# Focusing information (currently not implemented and probably never will be)
		#
		## FOCUSAXI *** NOT NEEDED ***
		#c30 = pyfits.Column(name='FOCUSAXI', format='1E', unit='m', 
						#array=numpy.array([0 for s in scanList]))
		## FOCUSTAN *** NOT NEEDED ***
		#c31 = pyfits.Column(name='FOCUSTAN', format='1E', unit='m', 
						#array=numpy.array([0 for s in scanList]))
		## FOCUSROT *** NOT NEEDED ***
		#c32 = pyfits.Column(name='FOCUSROT', format='1E', unit='deg', 
						#array=numpy.array([0 for s in scanList]))
		
		#
		# Weather information (currently not implemented)
		#
		## Ambient temperature *** UNKNOWN ***
		#c33 = pyfits.Column(name='TAMBIENT', format='1E', unit='C', 
						#array=numpy.array([0 for s in scanList]))
		## Air pressure *** UNKNOWN ***
		#c34 = pyfits.Column(name='PRESSURE', format='1E', unit='Pa', 
						#array=numpy.array([0 for s in scanList]))
		## Humidity *** UNKNOWN ***
		#c35 = pyfits.Column(name='HUMIDITY', format='1E', unit='%', 
						#array=numpy.array([0 for s in scanList]))
		## Wind speed *** UNKNOWN ***
		#c36 = pyfits.Column(name='WINDSPEE', format='1E', unit='m/s', 
						#array=numpy.array([0 for s in scanList]))
		## Wind direction *** UNKNOWN ***
		#c37 = pyfits.Column(name='WINDDIRE', format='1E', unit='deg', 
						#array=numpy.array([0 for s in scanList]))
		
		# Gather together all of the needed columns and figure out which ones
		# store the data and flag tables.  This information is needed later to
		# set the appropriate TDIM keywords.
		cs = []
                cs_new=[]
		dataIndex = 0
		#flagIndex = 0
		n = 1
		for i in xrange(1, 38):
			try:
				cs.append(eval('c%i' % i))
				if eval('c%i.name' %i) == 'DATA':
					dataIndex = n
				if eval('c%i.name' %i) == 'RFI':
					rfiIndex = n
                                        cs_new.append(eval('c%i' % i))
				if eval('c%i.name' %i) == 'RFI_MASK':
					rfiMaskIndex = n
                                        cs_new.append(eval('c%i' % i))                                        
				if eval('c%i.name' %i) == 'FLAGGED':
					flagIndex = n
				n += 1
			except NameError:
                                #if 20<i<28: print('sdfits: passing column name: c%s'%i)
				pass
                        
		colDefs = pyfits.ColDefs(cs)

		# Create the SINGLE DISH table and update its header
		sd = pyfits.new_table(colDefs)
		
		## Single disk keywords - order seems to matter
		sd.header.set('EXTNAME', 'SINGLE DISH', 'SDFITS table name', after='TFIELDS')
		sd.header.set('NMATRIX', 1, after='EXTNAME')
		sd.header.insert('TTYPE1', ('OBSERVER', self.observer, 'Observer name(s)'))
		sd.header.insert('TTYPE1', ('PROJID', self.project, 'Project name'))
		sd.header.insert('TTYPE1', ('TELESCOP', self.site.name, 'Telescope name'))
		x,y,z = self.site.getGeocentricLocation()
		sd.header.insert('TTYPE1', ('OBSGEO-X', x, '[m] Antenna ECEF X-coordinate'))
		sd.header.insert('TTYPE1', ('OBSGEO-Y', y, '[m] Antenna ECEF Y-coordinate'))
		sd.header.insert('TTYPE1', ('OBSGEO-Z', z, '[m] Antenna ECEF Z-coordinate'))
		
		sd.header['SPECSYS'] = ('LSRK', 'Doppler reference frame (transformed)')
		sd.header['SSYSOBS'] = ('TOPOCENT', 'Doppler reference frame of observation')
		sd.header['EQUINOX'] = (2000.0, 'Equinox of equatorial coordinates')
		sd.header['RADESYS'] = ('FK5', 'Equatorial coordinate system frame')
		
		## Data and flag table dimensionality
		sd.header.set('TDIM%i' % dataIndex, '(%i,%i,1,1)' % (self.nChan, self.nStokes), after='TFORM%i' % dataIndex)
		sd.header.set('TDIM%i' % flagIndex, '(%i,%i,1,1)' % (self.nChan, self.nStokes), after='TFORM%i' % flagIndex)
		
		## Data and flag table axis descriptions
		### Frequency
		sd.header.insert('TTYPE1', ('CTYPE1', 'FREQ', 'axis 1 is FREQ (frequency)'))
		sd.header.insert('TTYPE1', ('CDELT1', self.freq[0].chWidth))
		sd.header.insert('TTYPE1', ('CRPIX1', self.refPix))
		sd.header.insert('TTYPE1', ('CRVAL1', self.refVal))
		### Stokes
		sd.header.insert('TTYPE1', ('CTYPE2', 'STOKES', 'axis 2 is STOKES axis (polarization)'))
		if self.stokes[0] < 0:
			sd.header.insert('TTYPE1', ('CDELT2', -1.0))
		else:
			sd.header.insert('TTYPE1', ('CDELT2', 1.0))
		sd.header.insert('TTYPE1', ('CRPIX2', 1.0))
		sd.header.insert('TTYPE1', ('CRVAL2', float(self.stokes[0])))
		### RA
		sd.header.insert('TTYPE1', ('CTYPE3', 'RA', 'axis 3 is RA axis (pointing)'))
		sd.header.insert('TTYPE1', ('CRPIX3', 1.0))
		sd.header.insert('TTYPE1', ('CDELT3', -1.0))
		### Dec
		sd.header.insert('TTYPE1', ('CTYPE4', 'DEC', 'axis 4 is Dec. axis (pointing)'))
		sd.header.insert('TTYPE1', ('CRPIX4', 1.0))
		sd.header.insert('TTYPE1', ('CDELT4', 1.0))

		self.FITS.append(sd)
                
                if cs_new:
                        colDefs2 = pyfits.ColDefs(cs_new)                        
                        # Create the RFI table and update its header
                        sd2 = pyfits.new_table(colDefs2)
                        sd2.header.set('EXTNAME', 'RFI', 'SDFITS table name', after='TFIELDS')
                        sd2.header.set('NMATRIX', 1, after='EXTNAME')                        
                
                        ## Data and flag table dimensionality
                        sd2.header.set('TDIM%i' % rfiIndex, '(%i,%i,1,1)' % (self.nChan, self.nStokes),
                                       after='TFORM%i'%1)
                        sd2.header.set('TDIM%i' % rfiMaskIndex, '(%i,%i,1,1)' % (self.nChan, self.nStokes),
                                       after='TFORM%i'%2)
                        self.FITS.append(sd2)		

		self.FITS.flush()
