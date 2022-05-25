
from kamodo import Kamodo, kamodofy, gridify
from astropy.coordinates import SkyCoord
import sunpy.coordinates # needed to find sunpy coordinate frames
import numpy as np

import sunpy.map

from astropy.coordinates import SkyCoord


import forge

_frame_abbrev = {'HeliographicStonyhurst': 'HGS',
                 'Helioprojective': 'HPC',
                 'HeliographicCarrington': 'HGC'}

_frame_names = {v:k for k,v in _frame_abbrev.items()}

_earth_observers = ['HeliographicStonyhurst']

_available_frames = list(_frame_abbrev) + list(_frame_names)

def check_frame(frame):
    """See if this frame is supported"""

    if isinstance(frame, str):
        if frame not in _available_frames:
            raise NotImplementedError(f'{frame} not in {_available_frames}')
    else:
        for frame_ in frame:
            check_frames(frame_)


class SkyKamodo(Kamodo):
    """wrapper for Sky Coordinates"""
    def __init__(self,
                 to_frame, # can be list
                 from_frame, # can be list
                 from_units='arcsec',
                 representation_type='cartesian',
                 to_observer='earth',
                 from_observer='earth',
                 verbose=False,
                 **kwargs):
        """Register transformations between Sunpy Coordinate systems"""
        
        super(SkyKamodo, self).__init__(verbose=verbose, **kwargs)


        self._from_units = from_units
        self._to_frame = to_frame
        self._from_frame = from_frame
        self._from_observer = from_observer
        self._to_observer = to_observer
        self._representation_type = representation_type

        self.register_frames()

    @property
    def __doc__(self):
        return "Hey... Listen!"



    def register_frames(self):
        if isinstance(self._from_frame, str):
            if isinstance(self._to_frame, str):
                self.register_frame(self._from_frame, self._to_frame)
            else:
                for to_ in self._to_frame:
                    self.register_frame(self._from_frame, to_)
        else:
            if isinstance(self._to_frame, str):
                for from_ in self._from_frame: # assume iterable
                    self.register_frame(from_, self._to_frame)
            else:
                for from_ in self._from_frame:
                    for to_ in self._to_frame:
                        self.register_frame(from_, to_) 

    def register_frame(self, from_frame, to_frame):
        check_frame(from_frame)

        # in case abbreviation is used, get full name
        from_frame = _frame_names.get(from_frame, from_frame) 

        check_frame(to_frame)
        to_frame = _frame_names.get(to_frame, to_frame)

        if from_frame == to_frame:
            return

        if self.verbose:
            print(f'registering {from_frame} -> {to_frame}')
        from_abbrev = _frame_abbrev[from_frame]
        alpha_var = "alpha_{}".format(from_abbrev)
        delta_var = "delta_{}".format(from_abbrev)
        arg_units= {alpha_var: self._from_units,
                    delta_var: self._from_units,
                    't_unix': 's'}
        
        signature = [forge.arg(alpha_var),
                     forge.arg(delta_var),
                     forge.arg('t_unix')]
        
        
        To_frame = getattr(sunpy.coordinates.frames, to_frame)
        From_frame = getattr(sunpy.coordinates.frames, from_frame)

        if self._representation_type == 'cartesian':
            @kamodofy(units='km', arg_units=arg_units)
            @forge.sign(*signature)
            def transform(**kwargs):
                """Coordinate tranformer"""
                alpha_ = kwargs[alpha_var]
                delta_ = kwargs[delta_var]
                t_ = kwargs['t_unix']
                from_coord = SkyCoord(alpha_, delta_,
                                    unit=self._from_units,
                                    obstime=t_,
                                    observer=self._from_observer,
                                    frame=From_frame)

                if to_frame in _earth_observers:
                    # cannot use observer keyword on earth_observer frames
                    to_ = To_frame(obstime=t_)
                else:
                    to_ = To_frame(observer=self._to_observer, obstime=t_)
                    
                to_coord = from_coord.transform_to(to_)

                xvals = to_coord.cartesian.x.value
                yvals = to_coord.cartesian.y.value
                zvals = to_coord.cartesian.z.value
                return np.array([xvals, yvals, zvals])
            
            to_abbrev = _frame_abbrev[to_frame]
            reg_var = 'xvec_{}__{}'.format(to_abbrev, from_abbrev)
            transform.__name__ = reg_var
            
            docstring = f"Converts from {self._from_observer.capitalize()} {from_frame} " + \
             f"to {self._to_observer.capitalize()} {self._representation_type.capitalize()} {self._to_frame}"
            transform.__doc__ = docstring
        
        else:
            raise NotImplementedError("{} not supported".format(self._representation_type))
        
        self[reg_var] = transform







class SunMap(Kamodo):
    def __init__(self, image_filename, name='', verbose=False, **kwargs):
        """
        Initialize a kamodo object with an sunpy map object
        
        inputs:
            image_filename: path/to/fits/file
            name: name to use for function suffix/subscript
        """
        super(SunMap, self).__init__(verbose=verbose, **kwargs)
        self._image_filename = image_filename
        self.load_map()
        self._name = name
        
        self.register_coords()
    
    def load_map(self):
        self._sunmap = sunpy.map.Map(self._image_filename)

    def get_regname(self, prefix):
        if len(self._name) > 0:
            return f"{prefix}_{self._name}"
        else:
            return prefix

    def register_coords(self):

        @kamodofy(units='arcsec')
        def alpha(ivec):
            """pixel to longitude"""
            i, j = ivec
            return self._sunmap.wcs.pixel_to_world(i,j).spherical.lon.value
        
        self[self.get_regname('alpha')] = alpha

        @kamodofy(units='arcsec')
        def delta(ivec):
            """pixel to latitude"""
            i, j = ivec
            return self._sunmap.wcs.pixel_to_world(i,j).spherical.lat.value
        
        self[self.get_regname('delta')] = delta

        def xvec(ivec):
            """converts from pixels to Cartesian world coordinates.

            the output is normalized
            """
            i, j = ivec
            xvec_ = self._sunmap.wcs.pixel_to_world(i,j).cartesian
            x, y, z = xvec_.x.value, xvec_.y.value, xvec_.z.value
            return np.array([x,y,z])

        self[self.get_regname('xvec')] = xvec
    
        bbox = self.get_bbox()

        @kamodofy(arg_units=dict(alpha='arcsec', delta='arcsec'))
        @gridify(alpha = np.linspace(*bbox[0], 51), delta = np.linspace(*bbox[1], 53), order='A')
        def ivec(alphavec):
            """Gives the pixel coordinates for given longitude and latitude"""
            alpha_, delta_ = alphavec.T
            sky = SkyCoord(alpha_, delta_, unit='arcsec',
                       frame=self._sunmap.coordinate_frame)
            result = np.array(self._sunmap.wcs.world_to_pixel(sky))
            return np.ma.masked_less(result, 0).astype(int)
        
        ivec_name = self.get_regname('ivec')
        self[ivec_name] = ivec
        
        @kamodofy(units='')
        def I_px(ivec_px):
            """The image as a function of pixel indices
            input can be shape (2,3,4) or a 2-tuple of arrays of shape (3,4), (3,4)
            """
            i, j = ivec_px
            return self._sunmap.data[i, j] # not sure if this should be transposed or not
        
        self[self.get_regname('I_px')] = I_px
        
        self['i_px'] = f"I_px({ivec_name})" # composition

    def get_bbox(self):
        """get bounding box from alpha, delta (arseconds)"""
        delta_min, delta_max, alpha_min, alpha_max = np.nan*np.ones(4)
        imax, jmax = self._sunmap.data.shape
        alpha_func =  self[self.get_regname('alpha')]
        delta_func = self[self.get_regname('delta')]
        for i in [0, imax-1]:
            for j in [0, jmax-1]:
                alpha_ = alpha_func((i,j))
                delta_ = delta_func((i,j))
                alpha_min = np.nanmin([alpha_min, alpha_])
                alpha_max = np.nanmax([alpha_max, alpha_])
                delta_min = np.nanmin([delta_min, delta_])
                delta_max = np.nanmax([delta_max, delta_])

        return np.array([[alpha_min, alpha_max], [delta_min, delta_max]]).T

