
import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform  import rescale
import scipy,time,cProfile



N_angles=180
N_iter=20
activity_level=0.1;
original_image= shepp_logan_phantom();
original_image= rescale(activity_level * original_image,0.5)
Px= np.size(original_image,0)

def FP(image,azi_angles):
        
    F_proj = np.zeros((N_angles,Px))
 
    for i in range(N_angles):
        im_rot=scipy.ndimage.rotate(image,azi_angles[i],reshape=False)
        F_proj[i,:]= np.sum(im_rot, axis=0)
        
    return F_proj

def BP(projection,azi_angles):
    B_proj = np.zeros(original_image.shape)

    for i in range(N_angles):    
        B_proj += (1/N_angles)*scipy.ndimage.rotate(np.tile(projection[i,:],(Px,1)),-azi_angles[i],reshape=False)
    return B_proj



def my_MLEM():
    plt.ion()
    fig,axs=plt.subplots(2,3,figsize=(20,10))
    axs[0,0].imshow(original_image,cmap='Greys_r');    axs[0,0].set_title('Original Image')

    #Sinogram generation
    azi_angles= np.linspace(0.0,180.0,N_angles,endpoint=False)
    sinogram =FP(original_image,azi_angles)
    axs[0,1].imshow(sinogram,cmap='Greys_r');    axs[0,1].set_title('sinogram')

    

    mlem_rec  = np.ones(original_image.shape)
    sino_ones = np.ones(sinogram.shape)
    sens_image= BP(sino_ones,azi_angles)
    for iter in range(N_iter):
        fp= FP(mlem_rec,azi_angles)
        ratio= sinogram/(fp+0.000001)
        correction =BP(ratio,azi_angles)/sens_image

        axs[1,0].imshow(mlem_rec,cmap='Greys_r');    axs[1,0].set_title('MLEM recon')
        axs[1,1].imshow(fp,cmap='Greys_r');    axs[1,1].set_title('FP of recon')
        axs[0,2].imshow(ratio,cmap='Greys_r');    axs[0,2].set_title('Ratio of Sinogram')
        axs[1,2].imshow(correction,cmap='Greys_r');    axs[1,2].set_title('BP of ratio')

        mlem_rec=mlem_rec * correction
        axs[1,0].imshow(mlem_rec,cmap='Greys_r');    axs[1,0].set_title('Reconstructed image It=%d' % (iter+1))
        plt.show()
        plt.pause(0.5)
 
    plt.show(block=True)
    time.sleep(5)

    plt.close("all")    

 
if __name__=="__main__":
    with cProfile.Profile() as pr:
        my_MLEM()
    

    pr.dump_stats('my_mlem.prof')
