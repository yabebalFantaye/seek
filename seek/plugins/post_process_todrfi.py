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
                                filename)
        
        with h5py.File(filepath, "w") as fp:
            tod = self.ctx.tod_vx.data
            fp["data"] = tod
            
            if 'rfi_map' in dir(self.ctx.params):
                rfi = self.ctx.params.rfi_map
                fp["rfi_map"] = rfi
                rfi_mask = 100*np.abs(rfi/tod) > self.ctx.params.rfi_mask_frac  #frac is in percentage
                fp["gt_mask"] = np.bitwise_or(self.ctx.tod_vx.mask, rfi_mask)
                
                print("Writing TOD file {0}".format(filepath))
                r = np.divide(100.0*np.sum(rfi_mask), np.product(rfi_mask.shape))
                print("Fraction of RFI contaminated pixels = {0:.0f}%".format(r))
                print("")
            else:
                fp["gt_mask"] = self.ctx.tod_vx.mask
                
            fp["frequencies"] = self.ctx.frequencies
            fp["time"] = self.ctx.time_axis
            fp["ra"] = self.ctx.coords.ra
            fp["dec"] = self.ctx.coords.dec
            fp["ref_channel"] = self.ctx.ref_channel
            
            
    def __str__(self):
        return "postprocessing TOD"
