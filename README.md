# GRAVITY-Phasemaps
Script to create aberration maps for the VLTI instrument GRAVITY

The need and implementation is described in: [A&A 647, A59 (2021)](https://www.aanda.org/articles/aa/pdf/2021/03/aa40208-20.pdf)


## Set-up of class
The aberration maps have to be created for each wavelength individually and they are different for ATs & UTs. Both the wavelengths and the telescope have to be declared on initialization:
```python
wave = [2., 2.25, 2.5]
pm = Phasemaps(wave, 'AT')
```
If the phasemaps should be used for fitting they need to be created for each wavelength channel. In this case the wavelengths can be taken from a GRAVITY fits file:
```python
wave = fits.open(file)['OI_WAVELENGTH'].data['EFF_WAVE']
pm = Phasemaps(wave, 'AT')
```python

## Creating maps
The maps are then simply created and optionally plotted with:
```python
pm.createPhasemaps(nthreads=1)
pm.plotPhasemaps(pm.phasemap[1])
```
In case of many wavelengths multithreading is a good option, because the calculation will take ~7s per wavelength.

## Read out maps
For the read out on needs to have several parameters:
- ra, dec: star position relative to the fiber center, in mas
- northangle: rotation angle of the acquisition camera in rad (ideally taken from the header of a reduced file)
- dRa, dDec: misspointing of the fiber in mas (ideally taken from the header of a reduced file)

With these parameters one then calls:
```python
amplitude, phase, intensity = pm.readPhasemaps(ra, dec, northangle, dra, ddec)
```
Which gives the amplitude, phase-error and intensity at the given position in the fiber. Each output is an array with the size [number telescopes x length wave]


A full example is given in the available [jupyter notebook](https://github.com/widmannf/GRAVITY-Phasemaps/blob/main/CreatePhasemaps.ipynb).
