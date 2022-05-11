---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# SpacePy Tutorial -- An introduction to using MMS MEC data files

NASA's Magnetospheric Multiscale (MMS) mission includes a slightly unusual instrument. MEC is the _Magnetic Ephemeris and Coordinates_ and while it's officially an instrument, it's actually a supporting team that provides data on the position and attitude of the spacecraft as well as some derived magnetic field-related quantities.

This tutorial introduces a few key tools and techniques in the SpacePy and scientific Python ecosystem through illustrative use on MMS data.

*We note that MEC files with a major version number of 1 (i.e. v1.x.x) give the quaternions to rotate the frame.
MEC files with a major version number of >=2 (i.e. 2.x.x) give the quaternion to rotate the vector.*

*If using the version 1.x.x files the expected vector rotation can be found by taking the conjugates of each quaternion.*

```python
#start with some necessary module imports
import os
import numpy as np
import spacepy.datamodel as dm
import spacepy.toolbox as tb
import matplotlib.dates as mpd

#now some plotting setup, first we turn on "magic" inline plotting in the ipython notebook
%matplotlib inline
import matplotlib.pyplot as plt #imports plot library
import spacepy.plot as splot #gets spacepy plot tools and style sheets
```

First let's check whether we have the data file to work on...
If the named file isn't in this directory then we'll attempt to download it from the MMS SDC

```python
fname = 'mms1_mec_brst_l2_epht89d_20160301002916_v2.1.0.cdf'
if not os.path.isfile(fname):
    import urllib.request
    siteurl = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file='
    dataurl = ''.join([siteurl, fname])
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
quat_gse_to_eci = tb.quaternionConjugate(mmsdata['mms1_mec_quat_eci_to_gse'])
quat_gse_to_gsm = tb.quaternionMultiply(mmsdata['mms1_mec_quat_eci_to_gsm'], quat_gse_to_eci)
```

We know that GSE and GSM share the X-axis, so let's verify that the X axis is the same. We'll also look at the Z-axis, which should be different, though not radically so.

```python
print('X = {0}'.format(tb.quaternionRotateVector(quat_gse_to_gsm[0], [1,0,0])))
print('Y = {0}'.format(tb.quaternionRotateVector(quat_gse_to_gsm[0], [0,0,1])))
print('and the quaternion we used for this is\n({0})'.format(quat_gse_to_gsm[0]))
```

It's important to note here that there are two conventions for the representation of quaternions. Quaternions have three vector parts *(i,j,k)* and a scalar part *(w)*, thus the scalar part can be the first, or last, element; we store the scalar part in the final element of a 1x4 array.


### Comparing the computed positions
Since that all looks about right, let's rotate the first three elements of mms1_mec_r_gse into GSM and compare to mms1_mec_r_gsm

```python
myRgsm = tb.quaternionRotateVector(quat_gse_to_gsm[:3], mmsdata['mms1_mec_r_gse'][:3])
print(myRgsm - mmsdata['mms1_mec_r_gsm'][:3]) #should all be approx. zero
```

The differences are all better than machine precision. So, now that we've demonstrated how to use the quaternions to get from one system to another, let's get some actual data and transform it from satellite coordinates into a geophysical system.

Again, we'll need to grab a data file. For this example we'll use dfg magnetometer data.

```python
fname = 'mms1_dfg_brst_l1b_20160301002915_v1.19.0.cdf'
if not os.path.isfile(fname):
    import urllib
    siteurl = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file='
    dataurl = ''.join([siteurl, fname])
    urllib.urlretrieve(dataurl, filename=fname)



```

```python
#use SpacePy's datamodel for a convenient read -- a more direct interface to the CDF library is in spacepy.pycdf
fields = dm.fromCDF(fname)
```

```python
#extract the indices in the FIELDS file that are in the time-range given by the MEC file
inds = tb.tOverlapHalf(mmsdata['Epoch'], fields['Epoch'])
```

### Explore the data file, then visualize the B-field

```python
%matplotlib inline
#"magic" inline plotting in the ipython notebook
import matplotlib.pyplot as plt #imports plot library
```

```python
fields.tree()
```

```python
for key, value in fields['mms1_dfg_brst_bcs'].attrs.items():
    print('{0}: {1}'.format(key, value))
```

```python
npts = 3000
tt = mpd.date2num(mmsdata['Epoch'][:npts])
fieldstt = mpd.date2num(fields['Epoch'])
```

We're looking at a few seconds of finely sampled (burst mode) B-field data, so let's just do a linear interpolation on the samples to get the data at the sample times in the MEC file. The MEC burst files have 30 millisecond cadence; MEC survey mode files have 30 second cadence.

```python
fieldsBCS = np.empty([len(tt), 3])
fieldsBCS[:,0] = tb.interpol(tt, fieldstt, fields['mms1_dfg_brst_bcs'][:,0], left=np.nan, right=np.nan)
fieldsBCS[:,1] = tb.interpol(tt, fieldstt, fields['mms1_dfg_brst_bcs'][:,1], left=np.nan, right=np.nan)
fieldsBCS[:,2] = tb.interpol(tt, fieldstt, fields['mms1_dfg_brst_bcs'][:,2], left=np.nan, right=np.nan)
mag = tb.interpol(tt, fieldstt, fields['mms1_dfg_brst_bcs'][:,3])
```

### So new let's display the dfg data, in BCS coordinates, interpolated to the MEC timestamps

```python
fig = plt.figure()
ax = fig.add_subplot(111)
lObj = ax.plot(mmsdata['Epoch'][:npts], fieldsBCS)
ax.set_ylabel('B$_{BCS}$ [nT]')
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z'])


```

### And now we need to do the conversion of the field vector from BCS to GSM

Again we apply quaternion math, first making a quaternion to go from BCS to ECI, then multiply that to get a quetrnion that rotates our vector from BCS to GSM.

```python
quat_bcs_to_eci = tb.quaternionConjugate(mmsdata['mms1_mec_quat_eci_to_bcs'])
quat_bcs_to_gsm = tb.quaternionMultiply(mmsdata['mms1_mec_quat_eci_to_gsm'], quat_bcs_to_eci)

field_gsm = tb.quaternionRotateVector(quat_bcs_to_gsm[:npts], fieldsBCS)
```

### Let's plot the BCS and the GSM side by side

```python
fig = plt.figure(figsize=(14,3))
ax = fig.add_subplot(122)
lObj = ax.plot(mmsdata['Epoch'][:npts], field_gsm)
ax.set_ylabel('B$_{GSM}$ [nT]')
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z'])

ax = fig.add_subplot(121)
lObj = ax.plot(mmsdata['Epoch'][:npts], fieldsBCS)
ax.set_ylabel('B$_{BCS}$ [nT]')
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z'])
plt.subplots_adjust()
fig.suptitle('Day: {0}'.format(mmsdata['Epoch'][0].date().isoformat()))
```

### This looks reasonable as the rotation from BCS to GSM should largely be around the Z axis

We can do a further, simple check by rotating into DBCS, which should remove any periodicity (as it's the despun version of BCS)

```python
quat_bcs_to_eci = tb.quaternionConjugate(mmsdata['mms1_mec_quat_eci_to_bcs'])
quat_bcs_to_dbcs = tb.quaternionMultiply(mmsdata['mms1_mec_quat_eci_to_dbcs'], quat_bcs_to_eci)

field_dbcs = tb.quaternionRotateVector(quat_bcs_to_dbcs[:npts], fieldsBCS)
```

```python
fig = plt.figure(figsize=(14,3))
ax = fig.add_subplot(122)
lObj = ax.plot(mmsdata['Epoch'][:npts], field_dbcs)
ax.set_ylabel('B$_{DBCS}$ [nT]')
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z'])

ax = fig.add_subplot(121)
lObj = ax.plot(mmsdata['Epoch'][:npts], fieldsBCS)
ax.set_ylabel('B$_{BCS}$ [nT]')
fig.autofmt_xdate()
plt.legend(lObj, ['X','Y','Z'])
plt.subplots_adjust()
```

## So let's try a new day and compare with FGM data in different coordinates

These data should be publicly available, so if you don't have the file it will just download. This should also work with any later dates and versions.

```python
fname = 'mms1_fgm_brst_l2_20160131234434_v4.18.0.cdf'
if not os.path.isfile(fname):
    siteurl = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file='
    dataurl = ''.join([siteurl, fname])
    urllib.urlretrieve(dataurl, filename=fname)
fgmdata = dm.fromCDF(fname)
```

```python
fname = 'mms1_mec_brst_l2_epht89d_20160131234435_v2.1.0.cdf'
if not os.path.isfile(fname):
    siteurl = 'https://lasp.colorado.edu/mms/sdc/public/files/api/v1/download/science?file='
    dataurl = ''.join([siteurl, fname])
    urllib.urlretrieve(dataurl, filename=fname)
mecdata = dm.fromCDF(fname)
```

### Now we have the data file, let's quickly inspect what's in it

```python
fgmdata.tree()
```

And now we can run through the previous steps, interpolating the FGM data to the MEC timestamps.

```python
inds = tb.tOverlapHalf(mecdata['Epoch'], fgmdata['Epoch'])
tt = mpd.date2num(mecdata['Epoch'])
fgmtt = mpd.date2num(fgmdata['Epoch'][inds])

fgmBCS = np.empty([len(tt), 3])
fgmBCS[:,0] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_bcs_brst_l2'][inds,0], left=np.nan, right=np.nan)
fgmBCS[:,1] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_bcs_brst_l2'][inds,1], left=np.nan, right=np.nan)
fgmBCS[:,2] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_bcs_brst_l2'][inds,2], left=np.nan, right=np.nan)
fgmmagBCS = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_bcs_brst_l2'][inds,3], left=np.nan, right=np.nan)
```

```python
fgmGSM = np.empty([len(tt), 3])
fgmGSM[:,0] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gsm_brst_l2'][inds,0], left=np.nan, right=np.nan)
fgmGSM[:,1] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gsm_brst_l2'][inds,1], left=np.nan, right=np.nan)
fgmGSM[:,2] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gsm_brst_l2'][inds,2], left=np.nan, right=np.nan)
fgmmagGSM = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gsm_brst_l2'][inds,3], left=np.nan, right=np.nan)
```

```python
fgmDMPA = np.empty([len(tt), 3])
fgmDMPA[:,0] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_dmpa_brst_l2'][inds,0], left=np.nan, right=np.nan)
fgmDMPA[:,1] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_dmpa_brst_l2'][inds,1], left=np.nan, right=np.nan)
fgmDMPA[:,2] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_dmpa_brst_l2'][inds,2], left=np.nan, right=np.nan)
fgmmagDMPA = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_dmpa_brst_l2'][inds,3], left=np.nan, right=np.nan)
```

```python
fgmGSE = np.empty([len(tt), 3])
fgmGSE[:,0] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gse_brst_l2'][inds,0], left=np.nan, right=np.nan)
fgmGSE[:,1] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gse_brst_l2'][inds,1], left=np.nan, right=np.nan)
fgmGSE[:,2] = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gse_brst_l2'][inds,2], left=np.nan, right=np.nan)
fgmmagGSE = tb.interpol(tt, fgmtt, fgmdata['mms1_fgm_b_gse_brst_l2'][inds,3], left=np.nan, right=np.nan)
```

### Finally, we derive a quaternion to rotate the data from BCS to GSE and compare our rotated data to the GSE that's given in the file.

```python
quat_bcs_to_eci = tb.quaternionConjugate(mecdata['mms1_mec_quat_eci_to_bcs'])
q_bcs_to_gse = tb.quaternionMultiply(mecdata['mms1_mec_quat_eci_to_gse'], quat_bcs_to_eci)

myGSE = tb.quaternionRotateVector(q_bcs_to_gse, fgmBCS)
```

```python
#first we do the field that we rotated into GSE ourselves. Let's plot the x, y and z components in black
plt.plot(myGSE, 'k-')
#now let's overplot the GSE from the file in blue (x), green (y) and red (z). 
#Differences will be clear from the black lines peeking out behind the color
for col, ll in zip([0,1,2], ['b-','g-','r-']):
    plt.plot(fgmGSE[:,col], ll)
plt.ylabel('B [nT]')
```

## There is no visible difference between the results. Though the coordinate transforms done in the magnetometer files are using independent code, the results are consistent.

*We note here that the MEC files implement the full, and exact, MMS mission definitions for all coordinate systems. These should therefore be considered the reference transformations*
