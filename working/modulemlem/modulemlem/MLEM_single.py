#square uniform image as source, radon transform and backprojection
import numpy as np
import matplotlib.pyplot as plt
import scipy
import cProfile 
#import snakeviz






Px=100
sq_side= 40

N_iter=20
N_angles=200


def MLEM_rec():
    image = np.zeros((Px,Px))
    image[int((Px-sq_side)/2):int((Px+sq_side)/2),int((Px-sq_side)/2):int((Px+sq_side)/2) ]=1
    image[int((Px-sq_side/2)/2):int((Px+sq_side/2)/2),int((Px-sq_side/2)/2):int((Px+sq_side/2)/2) ]=0



    def FP(image):
        
        F_proj = np.zeros((N_angles,Px))
        angles = np.linspace(0., 180., N_angles)
        for i in range(N_angles):
            im_rot=scipy.ndimage.rotate(image,angles[i],reshape=False)
            F_proj[i,:]= np.sum(im_rot, axis=0)
        
        return F_proj

    def BP(projection):
        B_proj = np.zeros((Px,Px))
        angles = np.linspace(0., 180., N_angles)
        for i in range(N_angles):    
            B_proj += (1/N_angles)*scipy.ndimage.rotate(np.tile(projection[i,:],(Px,1)),-angles[i],reshape=False)
        return B_proj



    image_rec=[]
    sinogram=FP(image)
    norm=BP(np.ones((N_angles,Px)))
    image_rec.append(np.ones((Px,Px)))
    for i in range(30):
        image_rec.append(image_rec[-1]/norm*BP(sinogram/FP(image_rec[-1]))   )
   


    #SHOW ALL ITERATIONS
    # num_rows = int(np.ceil(np.sqrt(N_iter)))
    # num_cols = int(np.ceil(N_iter / num_rows))

    # fig, axs = plt.subplots(num_rows,num_cols,figsize=(10, 10))
    # axs = axs.flatten()
    # for i, ax in enumerate(axs):
    #     if i < N_iter:  
    #         ax.imshow(image_rec[i], vmin=0.0,vmax=1.)
    #         ax.set_title(f'k= {i}')
    #         ax.tick_params(axis='both', which='major', labelsize=1)
    #         ax.set_xticks([])  # Remove x ticks
    #         ax.set_yticks([])  # Remove y ticks

    #plt.tight_layout()
    #plt.show()
    # fig, axs = plt.subplots(1,2,figsize=(10, 10))
    # axs = axs.flatten()

    # axs[0].imshow(image, vmin=0.0,vmax=1.)
    # axs[0].set_title('Original image')
    # axs[1].imshow(image_rec[-1], vmin=0.0,vmax=1.)
    # axs[1].set_title('Reconstructed image')
    # for i, ax in enumerate(axs):
    #     ax.tick_params(axis='both', which='major', labelsize=1)
    #     ax.set_xticks([])  # Remove x ticks
    #     ax.set_yticks([])  # Remove y ticks

    # plt.show()

if __name__=="__main__":
    with cProfile.Profile() as pr:
        MLEM_rec()

    pr.dump_stats('example.prof')
   
    










