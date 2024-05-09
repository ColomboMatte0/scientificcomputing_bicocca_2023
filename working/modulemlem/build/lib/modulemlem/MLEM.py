
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib.animation as animation
import time





def MLEM():

    Px=100
    sq_side= 40
    N_angles=200

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

    def init():
        img.set_data(np.ones((Px,Px)))
        iteration_text.set_text('')
        return img, iteration_text

    def MLEM_update(i):
        
        grid=img.get_array()
        grid=grid/norm*BP(sinogram/(FP(grid)))
        img.set_array(grid)
        
        iteration_text.set_text('k= {}'.format(i))

        
        return img, iteration_text


    # #GRADIENT SQUARE
    # x = np.linspace(0, 1, Px)
    # y = np.linspace(0, 1, Px)
    # X, Y = np.meshgrid(x, y)
    # gradient = X * Y  
    # #Create a border of zeros around the gradient
    # border_width = int(0.2*Px)
    # image=np.zeros((Px,Px))
    # image[border_width:-border_width, border_width:-border_width] = gradient[border_width:-border_width, border_width:-border_width]
    # image=scipy.ndimage.gaussian_filter(image,0.5)

    ##HEART SHAPE
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    image= ((X**2 + Y**2 - 1)**3 - X**2 * Y**3) <= 0
    image = image.astype(np.float32)
    image= image[::-1, :]

    ## HOLLOW SQUARE
    # image=np.zeros((Px,Px))
    # image[int((Px-sq_side)/2):int((Px+sq_side)/2),int((Px-sq_side)/2):int((Px+sq_side)/2) ]=1
    # image[int((Px-sq_side/2)/2):int((Px+sq_side/2)/2),int((Px-sq_side/2)/2):int((Px+sq_side/2)/2) ]=0
    # #image=scipy.ndimage.gaussian_filter(image,0.5)

    sinogram=FP(image)
    sinogram[sinogram<0]=0
    sinogram = np.random.poisson(lam=sinogram)
    norm=BP(np.ones((N_angles,Px)))


    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(image)
    img = ax2.imshow(np.zeros((Px,Px)),vmin=0,vmax=1.0)
    iteration_text = ax2.text(0.5, 1.05, '', transform=ax2.transAxes, ha='center')
    anim = animation.FuncAnimation(fig, MLEM_update, init_func=init,
                                frames=100, interval=20)
    ax1.set_title('Original image')
    ax1.axis('off')
    ax2.axis('off')


    plt.show()

if __name__=="__main__":
    MLEM()
