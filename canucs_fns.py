import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.lines as mlines
from astropy.table import Table
import corner
import seaborn as sns
import os
# os.chdir('/path/to/git/clone/CANUCS-DR-Notebooks')

# if this doesn't work install the datasets package with pip install datasets
import datasets
from datasets import load_dataset

small_idlist = np.load('smallidlist.npz')['all_idx']

def flam_to_ujy(flam,lam):
    
    return ((flam*1e-19)*((lam**2.)*(1e6)) / (3e-5))

class CANUCSGalaxy:
    """Main class loading and plotting galaxies from CANUCS DR1"""
    
    def __init__(self, canucs_id = None):
        
        self.key = {'1':'macs0417','2':'a370','3':'macs0416','4':'macs1423','5':'macs1149'}
        self.key2 = {'1':'clu', '2':'ncf', '3':'nsf'}
        
        sns.set(font_scale= 1.4)
        sns.set_style('ticks')
        
        try:
            galaxy_index = small_idlist.tolist().index(canucs_id)
        except:
            print('galaxy not found in current dataset. please raise an issue.')
    
        ds = load_dataset("kiyer/canucs_test", 
                          split="train[%.0f:%.0f]" %(galaxy_index, galaxy_index+1)
                          # split=datasets.ReadInstruction("train", from_=10, to=20, unit="abs")
                         )
        print('loaded data')
        self.ds = ds
        
    def print_canucs_id(self):
        self.idx = self.ds['SOURCE'][0]
        s = str(self.idx)
        cluster = self.key[s[0]]
        field = self.key2[s[1]]
        self.fieldname = cluster+field
        print('Galaxy with CANUCS ID: '+str(self.idx)+ ' is in field '+ self.fieldname)
        
    def plot_photoz_posterior(self):

        self.zgrid = np.load('eazy_zgrid.npz')['zgrid']
        plt.axvline(self.ds['z_phot'][0])
        plt.plot(self.zgrid, self.ds['EAZY_ZPDF'][0])
        plt.xlim(0, self.ds['z_phot'][0]*2)
        plt.xlabel('redshift',fontsize=18)
        plt.ylabel('P(z)',fontsize=18); plt.yticks([])
        plt.title('Photo-z PDF for CANUCS ID: '+str(self.idx)+' (EAZY $\chi^2$: %.2f)' %self.ds['EAZY_CHI2'][0])
        plt.show()
        
    def plot_cutout(self, 
                    plt_per_row = 4, 
                    filt_list = ['F090W', 'F115W', 'F150W', 'F200W', 'F277W', 'F356W', 'F410M', 'F444W'],
                    filts_RGB = ['F444W','F277W','F150W'],
                    vmin=0, vmax_det_im = 1,vmax_cutouts=None,cmap='Spectral'):

        rows, cols = int(np.ceil((len(filt_list) + 3) / plt_per_row)), plt_per_row
        if vmax_cutouts == None:
            vmax_cutouts = np.amax(np.array(self.ds['SED_OBS'][0])*20)

        fig, axs = plt.subplots(rows,cols,figsize=(2*cols,2*rows),sharey=True,sharex=True)

        det_im = np.array(self.ds['det_im'][0])[0,:,:]
        segm = np.array(self.ds['segm'][0])[0,:,:]
        rgbmap = np.array(self.ds['rgb'][0])[0,:,:,:]

        for ax in axs.flat:
            ax.tick_params(length=0.,labelbottom=False,labelleft=False)

        ax_idx = 0
        ax = axs.flat[ax_idx] 
        ax.imshow(det_im,vmin=vmin,vmax=vmax_det_im,cmap='Greys_r',origin='lower',interpolation='nearest')

        ax_idx+=1
        ax = axs.flat[ax_idx] 
        ax.imshow(segm,cmap='Spectral',origin='lower',interpolation='nearest')

        ax_idx+=1
        xcoord,ycoord = 0.05,0.05
        rgb_spacer = 0.3
        ax = axs.flat[ax_idx] 
        ax.imshow(rgbmap,origin='lower',interpolation='nearest')
        ax.text(xcoord,ycoord,'{}'.format(filts_RGB[0]),c='r',ha='left', va='bottom', transform=ax.transAxes)
        ax.text(xcoord+(rgb_spacer),ycoord,'{}'.format(filts_RGB[1]),c='g',ha='left', va='bottom', transform=ax.transAxes)
        ax.text(xcoord+(2*rgb_spacer),ycoord,'{}'.format(filts_RGB[2]),c='b',ha='left', va='bottom', transform=ax.transAxes)
        ax.text(0.05,0.87,str(self.idx),fontsize=14,fontweight='bold',
                c='dodgerblue',ha='left', va='center', transform=ax.transAxes)
        ax_idx+=1

        for ii,filt in enumerate(filt_list):
            ax = axs.flat[ax_idx] 
            ax.imshow(np.array(self.ds[filt][0])[0,:,:],vmin=vmin,vmax=vmax_cutouts,cmap='Greys_r',origin='lower',interpolation='nearest')
            ax.text(0.05,0.87,filt,fontsize=14,c='dodgerblue',ha='left', va='center', transform=ax.transAxes)
            # ax.set_xlim([0,N])
            # ax.set_ylim([0,M])
            ax_idx+=1


        plt.subplots_adjust(wspace=0, hspace=0) 
        plt.show()
        
    def plot_sed_fits(self):
    
        labels = ['log M$_*$','log SFR$_{inst}$','$A_V$','log Z/Z$_\odot$']
        from matplotlib import rc
        rc('text',usetex=True)
        rc('text.latex', preamble=r'\usepackage{color}')
        import matplotlib.pyplot as plt

        if self.ds['bp_fits_exist'][0] == True:

            print('showing both Dense Basis and Bagpipes fits')

            tmpbpsamp = np.array(self.ds['corner_bp_samples'][0])
            dbtheta = np.array(self.ds['corner_db_theta'][0])
            dblikelihood = np.array(self.ds['corner_db_wts'][0])
            ranges = np.array(self.ds['corner_ranges'][0])

            lam_centers_bp = np.array(self.ds['lam_centers_bp'][0]) 
            lam_centers_db = lam_centers = np.array(self.ds['lam_centers_db'][0]) / 1e4
            lam_widths = np.array(self.ds['lam_widths_db'][0]) / 1e4
            obs_sed = np.array(self.ds['sed_obs'][0])
            obs_err = np.array(self.ds['err_obs'][0])

            db_sed = np.array(self.ds['sed_db'][0])
            db_lam = np.array(self.ds['lam_db'][0])
            db_spec = np.array(self.ds['spec_db'][0])
            db_lam_range = (db_lam > 0.4) & (db_lam < 5.)
            db_residuals = np.array(self.ds['residuals_db'][0])
            timeax_db = np.array(self.ds['timeax_db'][0])
            sfh_16_db = np.array(self.ds['sfh_16_db'][0])
            sfh_50_db = np.array(self.ds['sfh_50_db'][0])
            sfh_84_db = np.array(self.ds['sfh_84_db'][0])

            bp_sed = np.array(self.ds['sed_bp'][0])
            bp_lam = np.array(self.ds['lam_bp'][0])
            bp_spec = np.array(self.ds['spec_bp'][0])
            bp_lam_range = (bp_lam > 0.4) & (bp_lam < 5.)
            bp_residuals = np.array(self.ds['residuals_bp'][0])
            timeax_bp = np.array(self.ds['timeax_bp'][0])
            sfh_16_bp = np.array(self.ds['sfh_16_bp'][0])
            sfh_50_bp = np.array(self.ds['sfh_50_bp'][0])
            sfh_84_bp = np.array(self.ds['sfh_84_bp'][0])


            fig = plt.figure(figsize=(10, 10))
            fig_corner = corner.corner(tmpbpsamp,
                                      color='tomato',
                                      fig=fig,
                                      range=ranges,
                          plot_datapoints=False, 
                          fill_contours=True,quantiles=(0.16, 0.5, 0.84),
                          show_titles=True, smooth=1.0, title_kwargs={'fontsize':14}, hist_kwargs={'density':True},
                          levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)])

            fig_corner = corner.corner(dbtheta, 
                      weights=dblikelihood,
                      color='dodgerblue',
                      labels=labels, range=ranges,
                      plot_datapoints=False, 
                      fill_contours=True,quantiles=(0.16, 0.5, 0.84),
                      show_titles=True, smooth=1.0, title_kwargs={'fontsize':14},hist_kwargs={'density':True},
                      levels=[1 - np.exp(-(1/1)**2/2),1 - np.exp(-(2/1)**2/2)],
                      fig=fig_corner)

            ax_sed = fig.add_axes([0.7+0.2, 0.55+0.04, 0.775, 0.36])
            plt.errorbar(lam_centers_db, obs_sed,
                         yerr=obs_err, 
                         xerr=(lam_widths/2, lam_widths/2),
                         lw=0,elinewidth=2,capsize=3,marker='s',color='k')
            plt.plot(lam_centers_bp, bp_sed,'o', color='tomato',markersize=9)
            plt.plot(lam_centers_db, db_sed, 'o', color='dodgerblue',markersize=9)

            plt.yscale('log')
            plt.xscale('log'); plt.xlim(np.amin(lam_centers - lam_widths), np.amax(lam_centers + lam_widths))
            if (np.nanmin(obs_sed) < np.nanmax(obs_sed) / 100):
                plt.ylim(np.nanmax(obs_sed*1.4)/100,np.nanmax(obs_sed*2));
            tmpx, tmpy = plt.xlim(), plt.ylim()
            plt.plot(bp_lam[bp_lam_range], bp_spec[bp_lam_range], color='tomato',alpha=0.7)
            plt.plot(db_lam[db_lam_range], db_spec[db_lam_range], color='dodgerblue',alpha=0.7)
            plt.xlim(tmpx), plt.ylim(tmpy)
            plt.title('CANUCS ID: '+ str(self.idx)+' (in '+self.fieldname.upper()+')')
            plt.text(tmpx[1]*0.91, tmpy[0] *2.5, '$\chi^2_{DoF}: %.2f$ (Dense Basis)' %(self.ds['chi2_db'][0]), ha='right',va='bottom',fontsize=18,color='dodgerblue')
            plt.text(tmpx[1]*0.91, tmpy[0] *4.2, '$\chi^2_{DoF}: %.2f$ (Bagpipes)' %(self.ds['chi2_bp'][0]), ha='right',va='bottom',fontsize=18,color='tomato')
            plt.text(tmpx[1]*0.91, tmpy[0] *1.5, 'z$_{phot}$:%.2f' %(self.ds['z_phot'][0]), ha='right',va='bottom',fontsize=18)
            plt.ylabel(r'Flux Density [$\mu$Jy]')

            ax_res = fig.add_axes([0.7+0.2, 0.46+0.04, 0.775, 0.09])
            plt.axhline(0,color='k',alpha=0.3)
            plt.axhline(1,color='firebrick',linestyle='--',alpha=0.1)
            plt.axhline(-1,color='firebrick',linestyle='--',alpha=0.1)
            plt.plot(lam_centers_bp, bp_residuals,'o', color='tomato',markersize=9)
            plt.plot(lam_centers_db, db_residuals, 'o', color='dodgerblue',markersize=9)
            plt.xlabel('Wavelength [$\mu$m]'); plt.ylabel('$\chi$')

            ax_sfh = fig.add_axes([1.0+0.1, 0.1, 0.42+0.15, 0.3])
            plt.fill_between(timeax_bp, sfh_16_bp, sfh_84_bp, color='tomato',alpha=0.3)
            plt.plot(timeax_bp, sfh_50_bp, color='w',lw=7)
            plt.plot(timeax_bp, sfh_50_bp, color='tomato',lw=3, label='Bagpipes')

            plt.fill_between(timeax_db, sfh_16_db, sfh_84_db, color='dodgerblue',alpha=0.3)
            plt.plot(timeax_db, sfh_50_db, color='w',lw=7)
            plt.plot(timeax_db, sfh_50_db, color='dodgerblue',lw=3,label='Dense Basis')
            plt.legend(edgecolor='w')
            plt.xlim(0,np.amax(timeax_db))

            plt.xlabel('lookback time [Gyr]'); plt.ylabel('SFR(t) [M$_\odot$/yr]')

            plt.show()
            
    def get_sfh_dense_basis(self):
        timeax_db = np.array(self.ds['timeax_db'][0])
        sfh_16_db = np.array(self.ds['sfh_16_db'][0])
        sfh_50_db = np.array(self.ds['sfh_50_db'][0])
        sfh_84_db = np.array(self.ds['sfh_84_db'][0])
        return timeax_db, sfh_16_db, sfh_50_db, sfh_84_db

    def get_sfh_bagpipes(self):
        timeax_bp = np.array(self.ds['timeax_bp'][0])
        sfh_16_bp = np.array(self.ds['sfh_16_bp'][0])
        sfh_50_bp = np.array(self.ds['sfh_50_bp'][0])
        sfh_84_bp = np.array(self.ds['sfh_84_bp'][0])
        return timeax_bp, sfh_16_bp, sfh_50_bp, sfh_84_bp
    
    def get_sed_dense_basis(self):
        lam_centers_db = np.array(self.ds['lam_centers_db'][0]) / 1e4
        db_sed = np.array(self.ds['sed_db'][0])
        db_lam = np.array(self.ds['lam_db'][0])
        db_spec = np.array(self.ds['spec_db'][0])
        db_lam_range = (db_lam > 0.4) & (db_lam < 5.)
        db_residuals = np.array(self.ds['residuals_db'][0])
        return db_sed, lam_centers_db, db_lam, db_spec, db_residuals
    
    def get_sed_bagpipes(self):
        lam_centers_bp = np.array(self.ds['lam_centers_bp'][0]) 
        bp_sed = np.array(self.ds['sed_bp'][0])
        bp_lam = np.array(self.ds['lam_bp'][0])
        bp_spec = np.array(self.ds['spec_bp'][0])
        bp_residuals = np.array(self.ds['residuals_bp'][0])
        return bp_sed, lam_centers_bp, bp_lam, bp_spec, bp_residuals
    
    def list_keys(self):

        print('# Available keys are:\n')
        for fkey in self.ds.features:
            print(fkey)
            
    def get_key(self, key):
        if key[0] == 'F':
            return np.array(self.ds[key][0][0])
        else:
            return np.array(self.ds[key][0])