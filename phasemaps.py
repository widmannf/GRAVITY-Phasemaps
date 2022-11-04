import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from astropy import convolution
from joblib import Parallel, delayed

def print_status(number,total):
    number = number+1
    if number == total:
        print ("\rComplete: 100%")
    else:
        percentage = int((number/total)*100)
        print ("\rComplete: ", percentage, "%", end="")
        
        
class Phasemaps():
    def __init__(self, wave, tel):
        self.wave = wave
        if tel not in ['AT', 'UT']:
            raise ValueError('tel has to be "AT" or "UT"')
        self.tel = tel
        
    def phase_screen(self, zerpar, lam0=2.2, MFR=0.6308, stopB=8.0, stopS=0.96,
                     d1=8.0, dalpha=1., totN=1024, amax=100):
        """
        Simulate phase screens taking into account static aberrations.

        Parameters:
        -----------
        * Static aberrations in the pupil plane are described by 
        * low-order Zernicke polynomials
        * Their amplitudes are in units of micro-meter
        * A00 - A60

        * Static aberrations in the focal plane
        * B1m1 - B2p2

        * optical system
        MFR (float)   : sigma of fiber mode profile in units of dish radius
        stopB (float) : outer stop diameter in meters
        stopS (float) : inner stop diameter in meters

        * further parameters specify the output grid
        dalpha (float) : pixel width in the imaging plane in mas
        totN   (float) : total number of pixels in the pupil plane
        lam0   (float) : wavelength at which the phase screen is computed in
                         micro-meter
        d1     (float) : telescope to normalize Zernike RMS in m 
                         (UT=8.0, AT=1.82)
        amax   (float) : maximum off-axis distance in the maps returned
        """
        
        (A00, A1m1, A1p1, A2m2, A2p2, A20, A3m1, A3p1, A3m3,
         A3p3, A4m2, A4p2, A4m4, A4p4, A40,  A5m1, A5p1, A5m3,
         A5p3, A5m5, A5p5, A6m6, A6p6, A6m4, A6p4, A6m2, A6p2,
         A60, B1m1, B1p1, B20, B2m2, B2p2) = zerpar

        # --- coordinate scaling --- #
        lam0 = lam0*1e-6
        mas = 1.e-3 * (2.*np.pi/360) * 1./3600
        ext = totN*d1/lam0*mas*dalpha*dalpha
        du = dalpha/ext*d1/lam0

        # --- coordinates --- #
        ii = np.arange(totN) - (totN/2)
        ii = np.fft.fftshift(ii)

        # image plane
        a1, a2 = np.meshgrid(ii*dalpha, ii*dalpha)
        aa = np.sqrt(a1*a1 + a2*a2)

        # pupil plane
        u1, u2 = np.meshgrid(ii*du*lam0, ii*du*lam0)
        r = np.sqrt( u1*u1 + u2*u2 )
        t = np.angle(u1 + 1j*u2)

        # --- cut our central part --- #
        hmapN = int(amax/dalpha)
        cc = slice(int(totN/2)-hmapN, int(totN/2)+hmapN+1)
        if 2*hmapN > totN:
            print('Requested map sizes too large')
            return False

        # --- pupil function --- #
        pupil = r<(stopB/2.)
        if stopS > 0.:
            pupil = np.logical_and(r < (stopB/2.), r > (stopS/2.))

        # --- fiber profile --- #
        fiber = np.exp(-0.5*(r/(MFR*d1/2.))**2)
        if B1m1 != 0 or B1p1 != 0:
            fiber = np.exp(-0.5*((u1-B1m1)**2 +
                                 (u2-B1p1)**2)/(MFR*d1/2.)**2)

        # for higher-order focal plane aberrations we need to compute the 
        # fourier transform explicitly
        if np.any([B20, B2m2, B2p2] != 0):
            sigma_fib = lam0/d1/np.pi/MFR/mas
            sigma_ref = 2.2e-6/d1/np.pi/MFR/mas
            zernike = 0
            zernike += B1m1*2*(aa/sigma_ref)*np.sin(t)
            zernike += B1p1*2*(aa/sigma_ref)*np.cos(t)
            zernike += B20*np.sqrt(3.)*(2.*(aa/sigma_ref)**2 - 1)
            zernike += B2m2*np.sqrt(6.)*(aa/sigma_ref)**2*np.sin(2.*t)
            zernike += B2p2*np.sqrt(6.)*(aa/sigma_ref)**2*np.cos(2.*t)

            fiber = (np.exp(-0.5*(aa/sigma_fib)**2)
                     * np.exp(2.*np.pi/lam0*1j*zernike*1e-6))
            fiber = np.fft.fft2(fiber)

        # --- phase screens (pupil plane) --- #
        zernike = A00
        zernike += A1m1*2*(2.*r/d1)*np.sin(t)
        zernike += A1p1*2*(2.*r/d1)*np.cos(t)
        zernike += A2m2*np.sqrt(6.)*(2.*r/d1)**2*np.sin(2.*t)
        zernike += A2p2*np.sqrt(6.)*(2.*r/d1)**2*np.cos(2.*t)
        zernike += A20*np.sqrt(3.)*(2.*(2.*r/d1)**2 - 1)
        zernike += A3m1*np.sqrt(8.)*(3.*(2.*r/d1)**3 - 2.*(2.*r/d1))*np.sin(t)
        zernike += A3p1*np.sqrt(8.)*(3.*(2.*r/d1)**3 - 2.*(2.*r/d1))*np.cos(t)
        zernike += A3m3*np.sqrt(8.)*(2.*r/d1)**3*np.sin(3.*t)
        zernike += A3p3*np.sqrt(8.)*(2.*r/d1)**3*np.cos(3.*t)
        zernike += A4m2*np.sqrt(10.)*(2.*r/d1)**4*np.sin(4.*t)
        zernike += A4p2*np.sqrt(10.)*(2.*r/d1)**4*np.cos(4.*t)
        zernike += A4m4*np.sqrt(10.)*(4.*(2.*r/d1)**4 -3.*(2.*r/d1)**2)*np.sin(2.*t)
        zernike += A4p4*np.sqrt(10.)*(4.*(2.*r/d1)**4 -3.*(2.*r/d1)**2)*np.cos(2.*t)
        zernike += A40*np.sqrt(5.)*(6.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2 + 1)
        zernike += A5m1*2.*np.sqrt(3.)*(10*(2.*r/d1)**5 - 12*(2.*r/d1)**3 + 3.*2.*r/d1)*np.sin(t)
        zernike += A5p1*2.*np.sqrt(3.)*(10*(2.*r/d1)**5 - 12*(2.*r/d1)**3 + 3.*2.*r/d1)*np.cos(t)
        zernike += A5m3*2.*np.sqrt(3.)*(5.*(2.*r/d1)**5 - 4.*(2.*r/d1)**3)*np.sin(3.*t)
        zernike += A5p3*2.*np.sqrt(3.)*(5.*(2.*r/d1)**5 - 4.*(2.*r/d1)**3)*np.cos(3.*t)
        zernike += A5m5*2.*np.sqrt(3.)*(2.*r/d1)**5*np.sin(5*t)
        zernike += A5p5*2.*np.sqrt(3.)*(2.*r/d1)**5*np.cos(5*t)
        zernike += A6m6*np.sqrt(14.)*(2.*r/d1)**6*np.sin(6.*t)
        zernike += A6p6*np.sqrt(14.)*(2.*r/d1)**6*np.cos(6.*t)
        zernike += A6m4*np.sqrt(14.)*(6.*(2.*r/d1)**6 - 5.*(2.*r/d1)**4)*np.sin(4.*t)
        zernike += A6p4*np.sqrt(14.)*(6.*(2.*r/d1)**6 - 5.*(2.*r/d1)**4)*np.cos(4.*t)
        zernike += A6m2*np.sqrt(14.)*(15.*(2.*r/d1)**6 - 20.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2)*np.sin(2.*t)
        zernike += A6p2*np.sqrt(14.)*(15.*(2.*r/d1)**6 - 20.*(2.*r/d1)**4 - 6.*(2.*r/d1)**2)*np.cos(2.*t)
        zernike += A60*np.sqrt(7.)*(20.*(2.*r/d1)**6 - 30.*(2.*r/d1)**4 +12*(2.*r/d1)**2 - 1)

        phase = 2.*np.pi/lam0*zernike*1.e-6

        # --- transform to image plane --- #
        complexPsf = np.fft.fftshift(np.fft.fft2(pupil * fiber
                                                 * np.exp(1j*phase)))
        return complexPsf[cc, cc]/np.abs(complexPsf[cc, cc]).max()
    
    
    def createPhasemaps(self, nthreads=1, smooth=10):
        zernikefile = 'phasemap.npy'
        zer = np.load(zernikefile, allow_pickle=True).item()
        try:
            lwave = len(self.wave)
        except TypeError:
            self.wave = [self.wave]
            lwave = len(self.wave)

        if self.tel == 'UT':
            stopB = 8.0
            stopS = 0.96
            dalpha = 1
            totN = 1024
            d = 8
            amax = 100
            set_smooth = smooth

        elif self.tel == 'AT':
            stopB = 8.0/4.4
            stopS = 8.0/4.4*0.076
            dalpha = 1.*4.4
            totN = 1024
            d = 1.8
            amax = 100*4.4
            set_smooth = smooth

        kernel = convolution.Gaussian2DKernel(x_stddev=smooth)

        if nthreads == 1:
            all_pm = np.zeros((lwave, 4, 201, 201),
                            dtype=np.complex_)
            all_pm_denom = np.zeros((lwave, 4, 201, 201),
                                    dtype=np.complex_)
            print_status(-1, lwave)
            for wdx, wl in enumerate(self.wave):
                for GV in range(4):
                    zer_GV = zer['GV%i' % (GV+1)]
                    pm = self.phase_screen(zer_GV, lam0=wl, d1=d, stopB=stopB,
                                           stopS=stopS, dalpha=dalpha, totN=totN,
                                           amax=amax)
                    # if pm.shape != (201, 201):
                    #     print(pm.shape)
                    #     print('Need to convert to (201,201) shape')
                    #     pm = procrustes(pm, (201, 201), padval=0)
                    pm_sm = signal.convolve2d(pm, kernel, mode='same')
                    pm_sm_denom = signal.convolve2d(np.abs(pm)**2, 
                                                    kernel, mode='same')

                    all_pm[wdx, GV] = pm_sm
                    all_pm_denom[wdx, GV] = pm_sm_denom
                print_status(wdx, lwave)

        else:
            def multi_pm(lam):
                m_all_pm = np.zeros((4, 201, 201), dtype=np.complex_)
                m_all_pm_denom = np.zeros((4, 201, 201), dtype=np.complex_)
                for GV in range(4):
                    zer_GV = zer['GV%i' % (GV+1)]
                    pm = self.phase_screen(zer_GV, lam0=lam, d1=d, stopB=stopB,
                                           stopS=stopS, dalpha=dalpha,
                                           totN=totN, amax=amax)

                    if pm.shape != (201, 201):
                        print('Need to convert to (201,201) shape')
                        print(pm.shape)
                        pm = procrustes(pm, (201,201), padval=0)

                    pm_sm = signal.convolve2d(pm, kernel, mode='same')
                    pm_sm_denom = signal.convolve2d(np.abs(pm)**2, 
                                                    kernel, mode='same')
                    m_all_pm[GV] = pm_sm
                    m_all_pm_denom[GV] = pm_sm_denom
                return np.array([m_all_pm, m_all_pm_denom])

            res = np.array(Parallel(n_jobs=nthreads)(delayed(multi_pm)(lam)
                                                     for lam in self.wave))

            all_pm = res[:, 0, :, :, :]
            all_pm_denom = res[:, 1, :, :, :]
        savefile = ('Phasemap_%s_Smooth%i.npy'
                    % (self.tel, smooth))
        savefile2 = ('Phasemap_%s_Smooth%i_denom.npy'
                     % (self.tel, smooth))
        all_pm = all_pm
        all_pm_denom = all_pm_denom
        np.save(savefile, all_pm)
        np.save(savefile2, all_pm_denom)
        self.phasemap = all_pm
        self.phasemap_denom = all_pm_denom
        
        # Interpolate phasemaps in 4D grid
        x = np.arange(201)
        y = np.arange(201)
        itel = np.arange(4)
        iwave = np.arange(len(self.wave))
        points = (iwave, itel, x, y)
        
        amp_map = np.abs(all_pm)
        pha_map = np.angle(all_pm, deg=True)
        amp_map_denom = np.real(all_pm_denom)

        self.amp_map_int = interpolate.RegularGridInterpolator(points, amp_map)
        self.pha_map_int = interpolate.RegularGridInterpolator(points, pha_map)
        self.amp_map_denom_int = interpolate.RegularGridInterpolator(points, amp_map_denom)
    
    def plotPhasemaps(self, aberration_maps):

        """
        Plot phase- and amplitude maps.

        Parameters:
        ----------
        aberration_maps (np.array) : one complex 2D map per telescope
        fov (float)   : extend of the maps

        Returns: a figure object for phase-  and one for amplitude-maps.
        -------
        """

        def cut_circle(mapdat, amax):
            # cut a circtle from a quadratic map with radius=0.5*side length
            xcoord = np.linspace(-amax, amax, mapdat.shape[-1])
            yy, xx = np.meshgrid(xcoord, xcoord)
            rmap = np.sqrt(xx*xx + yy*yy)
            mapdat[rmap>amax] = np.nan
            return mapdat
        
        fov = 160
        if self.tel == 'AT':
            fov *= 4.4

        fs = plt.rcParams['figure.figsize']
        fig, ax = plt.subplots(2,2, sharex=True, sharey=True,
                               figsize=(fs[0], fs[0]))
        ax = ax.flatten()
        pltargsP = {'origin':'lower', 'cmap':'twilight_shifted',
                    'extent': [fov/2, -fov/2, -fov/2, fov/2],
                    'levels':np.linspace(-180, 180, 19, endpoint=True)}
        for io, img in enumerate(aberration_maps):
            img = np.flip(img, axis=1)[20:-20,20:-20]
            _phase = np.angle(img, deg=True)
            _phase = cut_circle(_phase, fov)
            imP = ax[io].contourf(_phase, **pltargsP)
            ax[io].set_aspect('equal')

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.25, 0.05, 0.5])
        fig.colorbar(imP, cax=cbar_ax, label='Phase [deg]')
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both',
                         top=False, bottom=False, left=False, right=False)
        plt.xlabel('Image plane x-cooridnate [mas]')
        plt.ylabel('Image plane y-cooridnate [mas]\n')
        plt.show()

        fig, ax = plt.subplots(2,2, sharex=True, sharey=True,
                               figsize=(fs[0], fs[0]))
        ax = ax.flatten()
        pltargsA = {'origin':'lower', 'vmin':0, 'vmax':1,
                    'extent': [fov/2, -fov/2, -fov/2, fov/2]}
        for io, img in enumerate(aberration_maps):
            img = np.flip(img, axis=1)[20:-20,20:-20]
            _amp = np.abs(img)
            _amp = cut_circle(_amp, fov)
            imA = ax[io].imshow(_amp, **pltargsA)
            ax[io].set_aspect('equal')

        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.25, 0.05, 0.5])
        fig.colorbar(imA, cax=cbar_ax, label='Fiber Throughput')
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', which='both',
                         top=False, bottom=False, left=False, right=False)
        plt.xlabel('Image plane x-cooridnate [mas]')
        plt.ylabel('Image plane y-cooridnate [mas]\n')
        plt.show()

    def rotation(self, ang):
        """
        Rotation matrix, needed for phasemaps
        """
        return np.array([[np.cos(ang), np.sin(ang)],
                         [-np.sin(ang), np.cos(ang)]])

    def readPhasemaps(self, ra, dec, northangle, dra, ddec):
        """
        Calculates coupling amplitude / phase for given coordinates
        ra,dec: RA, DEC position on sky relative to nominal
                field center = SOBJ [mas]
        dra,ddec: ESO QC MET SOBJ DRA / DDEC:
            location of science object (= desired science fiber position,
                                        = field center)
            given by INS.SOBJ relative to *actual* fiber position measured
            by the laser metrology [mas]
            mis-pointing = actual - desired fiber position = -(DRA,DDEC)
        north_angle: north direction on acqcam in degree
        northangle & dra,ddec should be taken from the GRAVITY fits file
        """
        wave = self.wave
        pm_pos = np.zeros((4, 2))
        readout_pos = np.zeros((4*len(wave),4))
        readout_pos[:, 0] = np.tile(np.arange(len(wave)), 4)
        readout_pos[:, 1] = np.repeat(np.arange(4), len(wave))

        for tel in range(4):
            pos = np.array([ra + dra[tel], dec + ddec[tel]])
            if self.tel == 'AT':
                pos /= 4.4
            pos_rot = np.dot(self.rotation(northangle[tel]), pos) + 100
            readout_pos[readout_pos[:, 1] == tel, 2] = pos_rot[1]
            readout_pos[readout_pos[:, 1] == tel, 3] = pos_rot[0]
            pm_pos[tel] = pos_rot

        cor_amp = self.amp_map_int(readout_pos).reshape(4, len(wave))
        cor_pha = self.pha_map_int(readout_pos).reshape(4, len(wave))
        cor_int_denom = self.amp_map_denom_int(readout_pos).reshape(4, len(wave))

        return cor_amp, cor_pha, cor_int_denom