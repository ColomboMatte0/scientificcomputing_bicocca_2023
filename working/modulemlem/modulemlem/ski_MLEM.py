
from skimage.data import shepp_logan_phantom
from skimage.transform  import radon, iradon, rescale

import matplotlib.pyplot as plt
import numpy as np
import cProfile 

#MLEM algorithm improved version using radon and iradon function from skimage version

N_angles=180
N_iter=20

def ski_MLEM():
    plt.ion()
    activity_level=0.1;
    original_image= shepp_logan_phantom();
    original_image= rescale(activity_level * original_image,0.5)
    fig,axs=plt.subplots(2,3,figsize=(20,10))
    axs[0,0].imshow(original_image,cmap='Greys_r');    axs[0,0].set_title('Original Image')

    #Sinogram generation
    azi_angles= np.linspace(0.0,180.0,N_angles,endpoint=False)
    sinogram =radon(original_image,azi_angles,circle=False)
    axs[0,1].imshow(sinogram.T,cmap='Greys_r');    axs[0,1].set_title('sinogram')

    

    mlem_rec  = np.ones(original_image.shape)
    sino_ones = np.ones(sinogram.shape)
    sens_image= iradon(sino_ones,azi_angles,circle=False,filter_name=None)
    for iter in range(N_iter):
        fp= radon(mlem_rec,azi_angles,circle=False)
        ratio= sinogram/(fp+0.000001)
        correction =iradon(ratio,azi_angles,circle=False,filter_name=None)/sens_image

        axs[1,0].imshow(mlem_rec,cmap='Greys_r');    axs[1,0].set_title('MLEM recon')
        axs[1,1].imshow(fp.T,cmap='Greys_r');    axs[1,1].set_title('FP of recon')
        axs[0,2].imshow(ratio.T,cmap='Greys_r');    axs[0,2].set_title('Ratio of Sinogram')
        axs[1,2].imshow(correction,cmap='Greys_r');    axs[1,2].set_title('BP of ratio')

        mlem_rec=mlem_rec * correction
        axs[1,0].imshow(mlem_rec,cmap='Greys_r');    axs[1,0].set_title('Reconstructed image It=%d' % (iter+1))
        plt.show()
        plt.pause(0.1)
 
    plt.show(block=True)


if __name__=="__main__":
    with cProfile.Profile() as pr:
        ski_MLEM()

    pr.dump_stats('ski_mlem.prof')
   
    










