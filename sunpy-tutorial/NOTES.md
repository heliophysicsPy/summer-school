# Planning Notes

* Follow same structure as previous tutorials: data download, data containers, coordinates
* Add some "extras" at the end to illustrate Dask/advantages of cloud computing
  * difference images for cutouts around some interesting region
  
  
## Dataset

* Look at the first CME observed by EUI: https://www.esa.int/ESA_Multimedia/Videos/2021/05/EUI_s_view_of_February_2021_CME/(lang)
  - It would be nice to get the more recent 2/22 CME, but that data does not seem to be publicly available yet: https://www.esa.int/ESA_Multimedia/Videos/2022/02/Solar_Orbiter_and_SOHO_s_view_of_a_giant_eruption_-_side_by_side
* Very favorable configuration of several different satellites: SDO, STEREO, SolO, PSP


## Outline

* NB1: Data download
  * Query data from (may want to pare this down somewhat)
    - AIA
    - HMI (synoptic, for field extrapolation)
    - STEREO A
    - GOES (show that the flare went off)
    - LASCO C2
    - EUI
    - EPD (timeseries, need to figure out which measurements are interesting here, if any; even if not, maybe just good to show that we can look at timeseries data from SolO)
  * Show off features of Fido
  * Construct simple queries (one image, one instrument for a time range)
  * Move to multi-instrument queries (maybe include HEK to show off metadata handling?)
  * Move to queries using external clients using SOAR
* NB2: Data containers
  * Show off features of Map
    - Plot an AIA image (use aiapy to do some additional processing here, maybe)
    - Do the usual API walk through, show derived properties and nice plotting/repr functionality
    - Plot all of the other images
    - Make a movie from our series of images from SolO
  * Show off features of TimeSeries
    - Plot GOES
    - Show API, plotting, repr
    - Plot EPD timeseries and point out usefullness with SolO data
* NB3: Coordinates
  - Use Albert's existing notebook to demonstrate the coords stack
  - Put into practice using the PFSS field extrapolation
    - Overplot the extrapolated field lines on top of images for different viewing angles
  - Reproject the STEREO image to AIA or the EUI image to the PSP observer location
* NB4: Image Processing Extras
  - Show how NDCube can be used to stack images in time (potentially prepare reproject images ahead of time?)
    - e.g. the EUI images we looked at previously
  - Use MGN filtering from sunkit image to bring out more structure in imager observations
  - Do running difference image of full-disk data using Dask
