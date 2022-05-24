
from astropy.coordinates import SkyCoord
import sunpy.coordinates # needed to find sunpy coordinate frames
from kamodo import Kamodo, kamodofy
import numpy as np

import forge

_frame_abbrev = {'HeliographicStonyhurst': 'HGS',
                 'Helioprojective': 'HPC',
                 'HeliographicCarrington': 'HGC'}

_earth_observers = ['HeliographicStonyhurst']

_available_frames = list(_frame_abbrev.keys()) + list(_frame_abbrev.values())

def check_frames(frame):
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
        check_frames(self._to_frame)
        check_frames(self._from_frame)

        if isinstance(self._from_frame, str):
            if isinstance(self._to_frame, str):
                self.register_frame(self._from_frame, self._to_frame)
            else:
                for to_ in to_frame:
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


