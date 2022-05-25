---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# SpacePy Tutorial -- An introduction to using MMS MEC data files

NASA's Magnetospheric Multiscale (MMS) mission includes a slightly unusual instrument. MEC is the _Magnetic Ephemeris and Coordinates_ and while it's officially an instrument, it's actually a supporting team that provides data on the position and attitude of the spacecraft as well as some derived magnetic field-related quantities.

This tutorial introduces a few key tools and techniques in the SpacePy and scientific Python ecosystem through illustrative use on MMS data.

*We note that MEC files with a major version number of 1 (i.e. v1.x.x) give the quaternions to rotate the frame.
MEC files with a major version number of >=2 (i.e. 2.x.x) give the quaternion to rotate the vector.*

*If using the version 1.x.x files the expected vector rotation can be found by taking the conjugates of each quaternion.*

### Setup
This tutorial uses leapsecond data that SpacePy normally maintains on a per-user basis. (To download or update this data on your own installation of SpacePy, use toolbox.update()).

For the Python in Heliophysics summer school, we have provided a shared directory with the normal SpacePy configuration and managed data. There are also other data files specific to the summer school so that data downloads don't need to be run. So we use a single directory containing all the data for this tutorial and also the `.spacepy` directory (normally in a user's home directory). We use an environment variable to point SpacePy at this directory before importing SpacePy; although we set the variable in Python, it can also be set outside your Python environment. Most users need never worry about this, but if you're not using this notebook in the summer school then set `is_pyhc = False` in the next cell before running it.

```python
import os
is_pyhc = True
if is_pyhc:
    tutorial_data = '/shared/jtniehof/spacepy_tutorial'  # All data for Python in Heliophysics summer school
    os.environ['SPACEPY'] = tutorial_data  # Use .spacepy directory inside this directory
else:
    tutorial_data = '.'
```

```python
#start with some necessary module imports
import urllib.request
import numpy as np
import spacepy.coordinates as spc
import spacepy.datamodel as dm
import spacepy.toolbox as tb
import matplotlib.dates as mpd

#now some plotting setup
import matplotlib.pyplot as plt  # imports plot library
import spacepy.plot as splot  # gets spacepy plot tools and style sheets
```

First let's check whether we have the data file to work on...
If the named file isn't in this directory then we'll attempt to download it from the MMS SDC

```python
dname = 'mms1_mec_srvy_l2_ephts04d_20201011_v2.2.1.cdf'
data_dir = os.path.join(tutorial_data, 'mms')
fname = os.path.join(data_dir, dname)
# make sure the output directory exists first
if not os.path.isdir(data_dir):
    os.path.mkdir(data_dir)
# and then if the files aren't present, download them
if not os.path.isfile(fname):
    siteurl = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file='
    dataurl = ''.join([siteurl, dname])
    urllib.request.urlretrieve(dataurl, filename=fname)
```

```python
mmsdata = dm.fromCDF(fname)
```

### Browse the data
To look at all the variables in a CDF loaded using SpacePy's datamodel we can just call the *tree* method. By setting the *verbose* keyword to **True** we can see the dimensions of each variable, and by setting the *attrs* keyword to **True** we'd be able to display the metadata associated with each variable.

Each file has a lot of variables, each with a lot of metadata, so to avoid filling the screen here we won't show the attributes.

```python
mmsdata.tree(verbose=True)
```

### To view metadata we just ask for the "attrs" of the variable
As noted above, the names of variable attributes can be visualized by *tree* when we add the *attrs* keyword.

```python
print(mmsdata['mms1_mec_quat_eci_to_gse'].attrs)
```

To get a "prettier" display of the attributes, we can iterate over the members and print the attribute name and contents.

```python
for key, value in mmsdata['mms1_mec_quat_eci_to_gse'].attrs.items():
    print('{0}: {1}'.format(key, value))
```

### So how do we use these quaternions?
The first test is to transform a set of known positions from one coordinate system to another. Here we'll go from GSE to GSM. We note that these values are not calculated with quaternions, but with rotation matrices in LANLGeoMag's CTrans routines. LANLGeoMag is available on github at: https://github.com/drsteve/LANLGeoMag. However, the SpacePy backend for coordinate transformation implements the same approach and calculates the results to the same accuracy with one major caveat: the MMS mission adopted the JPL DE421 model to specify the relative positions of all the bodies in the solar system, and this is how the Earth-Sun vector is derived. In SpacePy, there's currently only one method for calculating the Earth-Sun vector and it's a functional form rather than a fit to data (like the JPL DE models). However, in most cases the differences will be small.

So let's take the mms1_mec_r_gse position vectors and use the quaternions to rotate from GSE to GSM. Then we'll compare to the GSM values in the files to ensure we've done this correctly.

All the quaternions are given as eci_to*_targetsystem* . So to get from GSE to GSM, we need to take the following steps:
 - Take the conjugate of the ECI->GSE quaternion, giving a GSE->ECI quaternion.
 - Multiply the ECI->GSM quaternion by the GSE->ECI quaternion (Note the order is mult(rot2, rot1), where rot1 is the first transformation you want to make).
 - Use the resultant quaternion to rotate the vectors into the target system.

```python
quat_gse_to_eci = spc.quaternionConjugate(mmsdata['mms1_mec_quat_eci_to_gse'])
quat_gse_to_gsm = spc.quaternionMultiply(mmsdata['mms1_mec_quat_eci_to_gsm'], quat_gse_to_eci)
```

We know that GSE and GSM share the X-axis, so let's verify that the X axis is the same. We'll also look at the Z-axis, which should be different, though not radically so.

```python
# let's set numpy's print precision for tidiness
np.set_printoptions(precision=3)

print('X = {}'.format(spc.quaternionRotateVector(quat_gse_to_gsm[0], [1,0,0])))
print('Y = {}'.format(spc.quaternionRotateVector(quat_gse_to_gsm[0], [0,0,1])))
print('and the quaternion we used for this is\n({0})'.format(quat_gse_to_gsm[0]))
```

It's important to note here that there are two conventions for the representation of quaternions. Quaternions have three vector parts *(i,j,k)* and a scalar part *(w)*, thus the scalar part can be the first, or last, element.

We store the scalar part in the final element of a 1x4 array.

We can see that our unit vector along _X_ is basically [1, 0, 0] as 1e-17 is smaller than the floating point precision...
Similarly, our _Z_ unit vector has now been rotated so it has non-zero _Y_ and _Z_ components in GSM.

```python
print('{}'.format(np.finfo(float)))
```

### Comparing the computed positions
Since that all looks about right, let's rotate the first three elements of mms1_mec_r_gse into GSM and compare to mms1_mec_r_gsm

```python
print('The first three GSE position vectors [km] are\n{}\n'.format(mmsdata['mms1_mec_r_gse'][:3]))
print('The first three GSM position vectors [km] are\n{}\n'.format(mmsdata['mms1_mec_r_gsm'][:3]))
myRgsm = spc.quaternionRotateVector(quat_gse_to_gsm[:3], mmsdata['mms1_mec_r_gse'][:3])
print('The conversion error [km] is\n{}'.format(myRgsm - mmsdata['mms1_mec_r_gsm'][:3])) #should all be approx. zero
```

The differences are all extremely small. 1x10<sup>-11</sup> kilometers is 1 nanometer!

Let's try a different conversion, but using SpacePy's `coordinates`.

```python
import spacepy.time as spt
spc.DEFAULTS.set_values(use_irbem=False, itol=0)
tts = spt.Ticktock(mmsdata['Epoch'][-3:])
cc_ECI = spc.Coords(mmsdata['mms1_mec_r_eci'][-3:], 'ECI2000', 'car', ticks=tts)
cc_GEO = cc_ECI.convert('GEO', 'car')
print('The first three GEO position vectors [km] are\n{}\n'.format(mmsdata['mms1_mec_r_geo'][-3:]))
print('Using SpacePy instead of the supplied quaternions we get\n{}\n'.format(cc_GEO.data))
print('The conversion error [km] is\n{}'.format(cc_GEO.data - mmsdata['mms1_mec_r_geo'][-3:])) #should all be approx. zero
```

<!-- #region -->
Again, there's a difference here... 1x10<sup>-6</sup>km is 100 millimeters, which is probably due to differencing roundoff errors between the two calculations.

What if we start with the GSE position and convert to ECI(J2000) using both?

<details>
    <summary><b>(Click for answer)</b></summary>

```python
tts = spt.Ticktock(mmsdata['Epoch'][-3:])
cc_GSE = spc.Coords(mmsdata['mms1_mec_r_gse'][-3:], 'GSE', 'car', ticks=tts)
cc_ECI = cc_GSE.convert('ECI2000', 'car')
print('The first three ECI() position vectors [km] are\n{}\n'.format(mmsdata['mms1_mec_r_eci'][-3:]))
print('Using SpacePy instead of the supplied quaternions we get\n{}\n'.format(cc_ECI.data))
print('The conversion error [km] is\n{}'.format(cc_ECI.data - mmsdata['mms1_mec_r_eci'][-3:]))
```

The magnitude of error here is about 13km in the X position, and that conversion error doesn't change much with time. This is primarily due to the specific Earth-Sun position vector used in the MMS ephemeris calculation, which is different to that used by SpacePy. Of course, the relative differences between locations will be the same regardless of which transformation we use.

To see the difference made by using a lower accuracy transformation, change the backend to use the IRBEM library. You can do this either with a keyword argument when setting up your `Coords`, or by updating your deafult settings for `spacepy.coordinates`.

</details>

So, now that we've demonstrated how to use the quaternions to get from one system to another, let's get some actual data and transform it from satellite coordinates into a geophysical system.

Again, we'll need to grab a data file. For this example we'll use "FGM" magnetometer data.
<!-- #endregion -->

```python
dname = 'mms1_fgm_srvy_l2_20201011_v5.263.0.cdf'
fname = os.path.join(tutorial_data, 'mms', dname)
if not os.path.isfile(fname):
    siteurl = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file='
    dataurl = ''.join([siteurl, dname])
    urllib.request.urlretrieve(dataurl, filename=fname)
```

```python
#use SpacePy's datamodel for a convenient read -- a more direct interface to the CDF library is in spacepy.pycdf
# This is a big file, so let's use directly do just access what we want!
from spacepy import pycdf
fields = pycdf.CDF(fname)  # opens file, but doesn't read anything yet
# We should close this later, or open it using a context manager
# and just load what we need up front.
```

```python
#extract the indices in the FIELDS file that are in the time-range given by the MEC file
inds = tb.tOverlapHalf(mmsdata['Epoch'], fields['Epoch'][...])
```

### Explore the data file, then visualize the B-field

```python
tb.dictree(fields)
```

Note that data is provided here in four coordinate systems:
- BCS: Body Centered System
- DMPA: Despun Major Principal Axis
- GSE: Geocentric Solar Ecliptic
- GSM: Geocentric Solar Magnetospheric

For information about GSE and GSM, and other geophysical systems, see the [spacepy.coords](https://spacepy.github.io/coordinates.html) and [spacepy.ctrans](https://spacepy.github.io/ctrans.html) docs. The other two systems are MMS specific. BCS rotates with the spacecraft around the instantaneous spin axis (Z-axis). MPA - not used here - is similat to BCS, but the Z-axis is effectively the mean spin axis over a period of spin axis nutation. If there's no nutation, MPA and BCS should be the same. DMPA is _despun_ meaning that the X axis lies in the Sun-Earth plane.

Note that the Z-axis for BCS/MPA/DMPA is close to Earth's spin axis, so DMPA ends up being relatively close to GSE. Not close enough for many detailed analyses, but close enough that you can use it as a sanity check.

```python
bvar = 'mms1_fgm_b_dmpa_srvy_l2'
for key, value in fields[bvar].attrs.items():
    print('{0}: {1}'.format(key, value))
```

```python
mec_ticks = spt.Ticktock(mmsdata['Epoch'])
tt = mec_ticks.TAI
fieldstt = spt.Ticktock(fields['Epoch'][...]).TAI
```

We're looking at a day of survey mode B-field data, so let's assume it's fine to just do a linear interpolation on the samples to get the data at the sample times in the MEC file. The MEC survey files have 30 second cadence; MEC burst mode files have 30 millisecond cadence.

```python
fieldsDMPA = np.empty([len(mmsdata['Epoch']), 3])
fieldsDMPA[:,0] = np.interp(tt, fieldstt, fields[bvar][:,0], left=np.nan, right=np.nan)
fieldsDMPA[:,1] = np.interp(tt, fieldstt, fields[bvar][:,1], left=np.nan, right=np.nan)
fieldsDMPA[:,2] = np.interp(tt, fieldstt, fields[bvar][:,2], left=np.nan, right=np.nan)
mag = tb.interpol(tt, fieldstt, fields[bvar][:,3])
```

### So new let's display the FGM data, in DMPA coordinates, interpolated to the MEC timestamps

But first, before we forget, let's close the CDF file...

```python
fields.close()
# And back to the plotting
fig = plt.figure()
ax = fig.add_subplot(111)
lObj = ax.plot(mmsdata['Epoch'], fieldsDMPA)
ax.set_ylabel('B$_{DMPA}$ [nT]')
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z', 'Tot'])
```

### And now we need to do the conversion of the field vector from DMPA to GSM

Again we apply quaternion math, first making a quaternion to go from DMPA to ECI(J2000), then multiply that to get a quaternion that rotates our vector from DMPA to GSM.

```python
quat_dmpa_to_eci = spc.quaternionConjugate(mmsdata['mms1_mec_quat_eci_to_dmpa'])
quat_dmpa_to_gsm = spc.quaternionMultiply(mmsdata['mms1_mec_quat_eci_to_gsm'], quat_dmpa_to_eci)

field_gsm = spc.quaternionRotateVector(quat_dmpa_to_gsm, fieldsDMPA)
```

### Let's plot the DMPA and the GSM side by side

```python
timelims = spt.Ticktock(['2020-10-11T10:00:00', '2020-10-11T14:00:00']).UTC
fig = plt.figure(figsize=(14,3))
ax = fig.add_subplot(121)
lObj = ax.plot(mmsdata['Epoch'], fieldsDMPA)
ax.set_ylabel('B$_{DMPA}$ [nT]')
ax.set_xlim(timelims)
splot.applySmartTimeTicks(ax, timelims)
plt.legend(lObj, ['X','Y','Z'])
plt.subplots_adjust()
fig.suptitle('Day: {0}'.format(mmsdata['Epoch'][0].date().isoformat()))
ax.set_ylim([-15, 15])

ax = fig.add_subplot(122)
lObj = ax.plot(mmsdata['Epoch'], field_gsm)
ax.set_ylabel('B$_{GSM}$ [nT]')
ax.set_xlim(timelims)
ax.set_ylim([-15, 15])
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z'])
```

### Can we do _this_ transformation using SpacePy's Coords?

So far we've only transformed positions with `spacepy.coordinates.Coords`. So can we use `Coords` to transform arbitrary vectors between coordinate systems?

We'll test this by using the quaternions to get from DMPA to ECI(J2000), as the former is satellite-specific. Then we'll use `spacepy.coordinates.Coords` to rotate our magnetic field from ECI(J2000) to GSM.

```python
# For speed, let's set our "cache" to 5 minutes (300s) ...
# The accuracy required will depend on your application,
# and whether you are transforming to GEO!
spc.DEFAULTS.set_values(use_irbem=False, itol=300)

# Now rotate to ECI, make a Coords, then transform to GSM using that.
field_eci = spc.quaternionRotateVector(quat_dmpa_to_eci, fieldsDMPA)
field_ecispc = spc.Coords(field_eci, 'ECI2000', 'car', ticks=mec_ticks)
field_gsmspc = field_ecispc.convert('GSM', 'car')

# And plot as before
timelims = spt.Ticktock(['2020-10-11T10:00:00', '2020-10-11T14:00:00']).UTC
fig = plt.figure(figsize=(14,3))
ax = fig.add_subplot(121)
lObj = ax.plot(mmsdata['Epoch'], field_gsm)
ax.set_ylabel('B$_{GSM}$ [nT] -- Quat')
ax.set_xlim(timelims)
splot.applySmartTimeTicks(ax, timelims)
plt.legend(lObj, ['X','Y','Z'])
plt.subplots_adjust()
fig.suptitle('Day: {0}'.format(mmsdata['Epoch'][0].date().isoformat()))
ax.set_ylim([-15, 15])

ax = fig.add_subplot(122)
lObj = ax.plot(mmsdata['Epoch'], field_gsmspc.data)
ax.set_ylabel('B$_{GSM}$ [nT] -- Coord')
ax.set_xlim(timelims)
ax.set_ylim([-15, 15])
splot.applySmartTimeTicks(ax, timelims)
plt.legend(lObj, ['X','Y','Z'])
```

**Isn't _that_ reassuring!**

While we mostly use position vectors in examples for `Coords`, there's no reason (most of the time) that you can't use `Coords` to represent and transform arbitrary vectors between coordinate systems.

This won't work for coordinate systems like ENU (or the ground-based magnetometer favorite, NED) as they are _position-dependent_ transforms. There's [a discussion on out github](https://github.com/spacepy/spacepy/discussions/563) about this. Geodetic coordinates are another exception, as the use of an ellipsoid Earth means that a Cartesian representation makes no sense and the "local vertical" is not radially-outward. So again this is position dependent. Some common use cases may be supported in a future version of SpacePy, so watch this space.

### Interoperability with AstroPy coordinates

One last fun (?) task to show more package interoperability...
Imagine you have a special satellite-tracking telescope on the roof of your cabin at _Bear Mountain_. It's got the latest Alt-Az mount and will slew to whatever [Alt-Az coordinates](https://docs.astropy.org/en/stable/api/astropy.coordinates.AltAz.html) you punch in.
- Alt-Az stands for Altitude-Azimuth where _azimuth_ is an angle positive Eastward of North and _altitude_ is the elevation angle

```python
# This example is shamelessly modified from https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_obs-planning.html
import astropy.units as apu
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
mms_spc = spc.Coords(mmsdata['mms1_mec_r_eci'][0], 'ECI2000', 'car', ticks=mec_ticks[0])
mms_ap = mms_spc.to_skycoord()
bear_mountain = EarthLocation(lat=41.3*apu.deg, lon=-74*apu.deg, height=390*apu.m)
utcoffset = -4*apu.hour  # Eastern Daylight Time
time = mec_ticks.APT[0] - utcoffset
mmsaltaz = mms_ap.transform_to(AltAz(obstime=time, location=bear_mountain))
print("MMS's Altitude (angle), Azimuth = {:g}, {:g}".format(mmsaltaz.alt[0], mmsaltaz.az[0]))
```

So MMS should be viewable from our stated location at Bear Mountain, at the start of the day we were looking at.

Finally, there's nothing in this notebook (except the data) that's MMS-specific. Do you have vector-rotating quaternions? Do you have vectors you need to rotate between standard geophysical systems? No mission-specific support is required. We target _generic_ tools. And as interoperability between Python packages in the Heliophysics arena continues to increase, you can use the best tool for each particular job, flexibly. Building on top of the ecosystem saves time and effort, while reaping the benefits of well-tested code.
