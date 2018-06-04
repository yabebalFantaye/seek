# SEEK is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# SEEK is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with SEEK.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Feb 6, 2015

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

from ivy.plugin.base_plugin import BasePlugin
import h5py
import os
import numpy as np

#
from sdfits import SD as sdfits
import ephem

"""
Offset in days between standard Julian day and Dublin Julian day.
"""
DJD_OFFSET = 2415020.0

def geo2ecef(lat, lon, elev):
	"""
	Convert latitude (rad), longitude (rad), elevation (m) to earth-
	centered, earth-fixed coordinates.
	"""
	
	WGS84_a = 6378137.000000
	WGS84_b = 6356752.314245
	N = WGS84_a**2 / numpy.sqrt(WGS84_a**2*numpy.cos(lat)**2 + WGS84_b**2*numpy.sin(lat)**2)
	
	x = (N+elev)*numpy.cos(lat)*numpy.cos(lon)
	y = (N+elev)*numpy.cos(lat)*numpy.sin(lon)
	z = ((WGS84_b**2/WGS84_a**2)*N+elev)*numpy.sin(lat)
	
	return (x, y, z)


def ecef2geo(x, y, z):
	"""
	Convert earth-centered, earth-fixed coordinates to (rad), longitude 
	(rad), elevation (m) using Bowring's method.
	"""
	
	WGS84_a = 6378137.000000
	WGS84_b = 6356752.314245
	e2 = (WGS84_a**2 - WGS84_b**2) / WGS84_a**2
	ep2 = (WGS84_a**2 - WGS84_b**2) / WGS84_b**2
	
	# Distance from rotation axis
	p = numpy.sqrt(x**2 + y**2)
	
	# Longitude
	lon = numpy.arctan2(y, x)
	p = numpy.sqrt(x**2 + y**2)
	
	# Latitude (first approximation)
	lat = numpy.arctan2(z, p)
	
	# Latitude (refined using Bowring's method)
	psi = numpy.arctan2(WGS84_a*z, WGS84_b*p)
	num = z + WGS84_b*ep2*numpy.sin(psi)**3
	den = p - WGS84_a*e2*numpy.cos(psi)**3
	lat = numpy.arctan2(num, den)
	
	# Elevation
	N = WGS84_a**2 / numpy.sqrt(WGS84_a**2*numpy.cos(lat)**2 + WGS84_b**2*numpy.sin(lat)**2)
	elev = p / numpy.cos(lat) - N
	
	return lat, lon, elev

class Station(ephem.Observer)
	def __init__(self, name, lat, long, elev, id='', antennas=None, interface=None):
        
		self.init_info(name, id=id, antennas=antennas, interface=interface)
		ephem.Observer.__init__(self)
		
		self.lat = lat * numpy.pi/180.0
		self.long = long * numpy.pi/180.0
		self.elev = elev
		self.pressure = 0.0
        
	def init_info(self, name, id='', antennas=None, interface=None):
		self.name = name
		self.id = id
		
		if antennas is None:
			self.antennas = []
		else:
			self.antennas = antennas
			
		if interface is None:
			self.interface = []
		else:
			self.interface = interface

	def getObserver(self, date=None, JD=False):
		"""
		Return a ephem.Observer object for this site.
		"""
		
		oo = ephem.Observer()
		oo.lat = 1.0*self.lat
		oo.long = 1.0*self.long
		oo.elev = 1.0*self.elev
		oo.pressure = 0.0
		if date is not None:
			if JD:
				# If the date is Julian, convert to Dublin Julian Date 
				# which is used by ephem
				date -= DJD_OFFSET
			oo.date = date
			
		return oo
            
	def getAIPYLocation(self):
		"""
		Return a tuple that can be used by AIPY for specifying a array
		location.
		"""
		
		return (self.lat, self.long, self.elev)
		
	def getGeocentricLocation(self):
		"""
		Return a tuple with earth-centered, earth-fixed coordinates for the station.
		"""
		
		return geo2ecef(self.lat, self.long, self.elev)
    


class Plugin(BasePlugin):
    """
    Writes the data, mask and frequencies of the current iteration to disk. Can
    be used for closer analysis of the masking (sum threshold). Output is
    written to the current folder using the same filename as the first input
    filename (may overwrite the original file if not being careful)
    """

    def __call__(self):
        if not self.ctx.params.store_intermediate_result:
            return
        
        filename = os.path.basename(self.ctx.file_paths[0])
        filepath = os.path.join(self.ctx.params.post_processing_prefix,
                                filename.split('.')[0],'.fits')

        #get all necessary data
        tod = self.ctx.tod_vx.data
        #fp["ra"] = self.ctx.coords.ra
        #fp["dec"] = self.ctx.coords.dec
        #fp["ref_channel"] = self.ctx.ref_channel
        
        if 'rfi_map' in dir(self.ctx.params):
            rfi = self.ctx.params.rfi_map
            rfi_mask = 100*np.abs(rfi/tod) > self.ctx.params.rfi_mask_frac  #frac is in percentage
            rfi_mask = np.bitwise_or(self.ctx.tod_vx.mask, rfi_mask)

            print("Writing TOD file {0}".format(filepath))
            r = np.divide(100.0*np.sum(rfi_mask), np.product(rfi_mask.shape))
            print("Fraction of RFI contaminated pixels = {0:.0f}%".format(r))
            print("")
        else:
            rfi_mask = self.ctx.tod_vx.mask
            
        #telescope data
        lon = self.ctx.params.telescope_longitude
        lat = self.ctx.params.telescope_latitude
        elv = self.ctx.params.beam_elevation
        
        #Start writing sdfits file
        sd = sdfits(filepath, refTime=self.ctx.batch_start_date,
                        verbose=False,
                        memmap=None,
                        clobber=self.ctx.params.overwrite)

        
        site = Station('belein',lat,lon,elv)
        sd.setsite(site)
        sd.setStoke(['XX'])
        sd.setFrequency(self.ctx.frequencies)
        sd.setObserver('UNKNOWN')

        #integration time 
        intTime = self.ctx.params.integration_time
        beam = ['TOD','RFI','RFI_MASK']

        for ix, obsTime in enumerate(self.ctx.time_axis):
            #define different data as different beams
            tod_ix = np.stack([tod[:,ix],rfi[:,ix],rfi_mask[:,ix]],axis=0)                            
            sd.addDataSet(self, obsTime,
                          intTime,
                          beam,
                          tod_ix,
                          pol='XX'):
        sd.write()
        sd.close()
                                       
    def __str__(self):
        return "postprocessing TOD and writing to SDFITS"
