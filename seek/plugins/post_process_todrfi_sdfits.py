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
from seek.plugins.sdfits import SD as sdfits
import seek.plugins.jdutil as jdutil
import seek.plugins.astro as astro
import ephem
import datetime


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
	N = WGS84_a**2 / np.sqrt(WGS84_a**2*np.cos(lat)**2 + WGS84_b**2*np.sin(lat)**2)
	
	x = (N+elev)*np.cos(lat)*np.cos(lon)
	y = (N+elev)*np.cos(lat)*np.sin(lon)
	z = ((WGS84_b**2/WGS84_a**2)*N+elev)*np.sin(lat)
	
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
	p = np.sqrt(x**2 + y**2)
	
	# Longitude
	lon = np.arctan2(y, x)
	p = np.sqrt(x**2 + y**2)
	
	# Latitude (first approximation)
	lat = np.arctan2(z, p)
	
	# Latitude (refined using Bowring's method)
	psi = np.arctan2(WGS84_a*z, WGS84_b*p)
	num = z + WGS84_b*ep2*np.sin(psi)**3
	den = p - WGS84_a*e2*np.cos(psi)**3
	lat = np.arctan2(num, den)
	
	# Elevation
	N = WGS84_a**2 / np.sqrt(WGS84_a**2*np.cos(lat)**2 + WGS84_b**2*np.sin(lat)**2)
	elev = p / np.cos(lat) - N
	
	return lat, lon, elev

class Station(ephem.Observer):
	def __init__(self, name, lat, long, elev, id='', antennas=None, interface=None):
        
		self.init_info(name, id=id, antennas=antennas, interface=interface)
		ephem.Observer.__init__(self)
		
		self.lat = lat * np.pi/180.0
		self.long = long * np.pi/180.0
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
                                filename.split('.')[0])+'.fits'

        flag_dir=os.path.join(self.ctx.params.post_processing_prefix,'flagged')
        if not os.path.exists(flag_dir):
                os.mkdir(flag_dir)
        filepath_flag = os.path.join(flag_dir,filename.split('.')[0])+'.fits'
        
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
            rfi = rfi_mask*1.0
            
        # #----- first write hdf5 files -----
        # with h5py.File(filepath, "w") as fp:
        #     tod = self.ctx.tod_vx.data
        #     fp["data"] = tod
            
        #     if 'rfi_map' in dir(self.ctx.params):
        #         fp["rfi_map"] = rfi
        #         fp["gt_mask"] = np.bitwise_or(self.ctx.tod_vx.mask, rfi_mask)
        #     else:
        #         fp["gt_mask"] = self.ctx.tod_vx.mask
                
        #     fp["frequencies"] = self.ctx.frequencies
        #     fp["time"] = self.ctx.time_axis
        #     fp["ra"] = self.ctx.coords.ra
        #     fp["dec"] = self.ctx.coords.dec
        #     fp["ref_channel"] = self.ctx.ref_channel

        #----- write sdfits files -----            
        #telescope data
        lon = self.ctx.params.telescope_longitude
        lat = self.ctx.params.telescope_latitude
        elv = self.ctx.params.telescope_elevation
        
        #Start writing sdfits file
        sd_rfi = sdfits(filepath, refTime=self.ctx.strategy_start,
                        verbose=False,
                        memmap=None,
                        clobber=self.ctx.params.overwrite)

        sd_flag = sdfits(filepath_flag, refTime=self.ctx.strategy_start,
                        verbose=False,
                        memmap=None,
                        clobber=self.ctx.params.overwrite)

        #integration time 
        intTime = self.ctx.params.integration_time
        beam = ['TOD']
        beam = [0]
        
        for sdname, sd in zip(['rfi','flag'],[sd_rfi,sd_flag]):
                if sdname == 'rfi':
                        print('Saving RFI sdfits..')
                else:
                        print('Saving FLAG sdfits')
                        
                site = Station('Arecibo',lat,lon,elv)
                sd.setSite(site)
                sd.setStokes(['XX'])
                sd.setFrequency(self.ctx.frequencies)
                sd.setObserver('UNKNOWN')

        
                for ix, dt in enumerate(self.ctx.time_axis):
                        obsTime_dt = self.ctx.strategy_start + datetime.timedelta(seconds=int(dt*3600))
                        obsTime_jd = jdutil.datetime_to_jd(obsTime_dt)
                        # if ix<10:
                        #         print(obsTime_dt, obsTime_jd,type(obsTime_dt), type(obsTime_jd))             

                        if sdname == 'rfi':
                                sd.addDataSet(obsTime_jd,
                                  intTime,
                                  beam,
                                  np.expand_dims(tod[:,ix],axis=0),
                                  rfi=np.expand_dims(rfi[:,ix],axis=0),
                                  rfi_mask=np.expand_dims(rfi_mask[:,ix],axis=0)
                                )
                        else:
                                sd.addDataSet(obsTime_jd,
                                  intTime,
                                  beam,
                                  np.expand_dims(tod[:,ix],axis=0),
                                )
                    
                sd.write()
                sd.close()
                                       
    def __str__(self):
        return "postprocessing TOD and writing to SDFITS"
