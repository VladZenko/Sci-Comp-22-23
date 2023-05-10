import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
import cv2




#======================================================================================
#|                                     Lensing                                        |
#======================================================================================


def lens(source, rc, eps, chnls=3, rng=1):
    '''
    This function lenses the input image using symmetric lense (dot object)
    at the centre of the image

    ----------------------------------------------------------------------
    Input:
    
    1) source - Square image, RGB, pixel valuse 0 -> 255
    2) rc - core radius in reduced coordinates
    3) eps - eccentricity of the lens
    4) chnls - amount of channels (3rd dimension) in the image array (3 for RGB, 1 for grayscale)
    5) rng - range of reduced r. Default is set to 1 as in the validation test,
             so that reduced plane goes from -1 to 1

    ----------------------------------------------------------------------
    Output:

    1) lensed image object

    '''
    
    # check if the input is square
    if np.shape(source)[0] == np.shape(source)[1]:
        size = np.shape(source)[0]
    else:
        print("Please, provide square image")
        exit()
    
    # set up an empty array to store lensed image
    img_lens = np.zeros([size, size, chnls])
    pixel_wdth = 2*rng/(size)
    
    
    # pixel indices (used to get reduced coordinates)
    i = np.arange(0, size, 1)
    j = np.arange(0, size, 1)
    
    # meshgrid of reduced x, y (denoted as r1, r2 respectively)
    r1 = 2*rng*i/(size) - rng + pixel_wdth/2
    r2 = 2*rng*j/(size) - rng + pixel_wdth/2
    r1g, r2g = np.meshgrid(r1, r2)
    
    # use lens equation to get positions on image_s
    s1 = r1g - ((1 - eps)*r1g)/np.sqrt(rc**2 + (1 - eps)*r1g**2 + (1 + eps)*r2g**2)
    s2 = r2g - ((1 + eps)*r2g)/np.sqrt(rc**2 + (1 - eps)*r1g**2 + (1 + eps)*r2g**2)
    
    # find corresponding pixel in the original image
    idx1 = np.floor((s1 + rng)/pixel_wdth)
    idx1 = idx1.astype(int).transpose()
    idx2 = np.floor((s2 + rng)/pixel_wdth) 
    idx2 = idx2.astype(int).transpose()  # and tranpose to match directions
    
    # map the source image data at the indices to the lensed image array
    img_lens[:, :, :] += source[idx1, idx2, :]        
    
    # return image
    return img_lens



#======================================================================================
#|                            Galaxy Cluster Generator                                |
#======================================================================================



def gal_dist(img_size, N_gal, mj_ax, ellipticity, seed="Y"):

    '''
    This function generates a distribution of ellipsoidal galaxies on a source plane of specified size.
    Custom number of galaxies, maximum major axis size factor, ellipticity and seed can be
    supplied by the user.

    ----------------------------------------------------------------------
    Input:
    
    1) img_size - length of the source plane side in pixels (source has to be square)
    2) N_gal - amount of galaxies generated
    3) mj_ax - major axis of the ellipse factor which is multiplied by minimum axis size and added to it.
    4) ellipticity - eccentricity parameter of the ellipses (from 0 to 1)
    5) seed - specify whether the generation should be seeded, i.e. consistent results over many runs. Default set to "Y" (yes)


    ----------------------------------------------------------------------
    Output:

    1) Image object with distribution of elliptical galaxies

    '''
    # seed the result (conserved over many runs) if specified
    if seed=='Y':
        np.random.seed(1234)
    else:
        pass

    # host plane for galaxies
    src = np.zeros((img_size, img_size, 3), dtype=np.float64)

    # randomly generated centers of galaxies
    cntrs = (np.random.randint(img_size, size=(N_gal,2)))
    cntrs = np.vstack(np.append(cntrs, np.array([(img_size/2), int(img_size/2)]))) # stack into 2d array
    cntrs = np.reshape(cntrs, (-1, 2))

    # random rotation angles for ellipses
    angles = (np.random.randint(360, size=(N_gal+1)))
    angles = angles*(np.pi/180)

    # factors which determine size of a single galaxy
    axs_fctr = (np.random.randint(mj_ax, size=(N_gal+1)))
    aa = ((img_size/25) +  np.abs(axs_fctr)*(img_size/25)) # semi-major axis

    # random colours
    rgb = np.random.rand(int(N_gal+1), 3)

    Eps = ellipticity
    


    for i in range(len(cntrs[:])):
        
        cntr = cntrs[i]
        th = angles[i]

        a = aa[i]
        c = Eps*a
        b = (a**2 - c**2)**0.5 # semi minor axis

        R = rgb[i][0]
        G = rgb[i][1]
        B = rgb[i][2]

        # meshgrid of coordinates to plot an ellipse in
        x, y = np.meshgrid(np.linspace(0,img_size,img_size)-cntr[0], np.linspace(0,img_size,img_size)-cntr[1])

        # rotate by the random angle
        x_ang = x[:]*np.cos(th) + y[:]*np.sin(th)
        y_ang = x[:]*np.sin(th) - y[:]*np.cos(th)


        e = 0.075  # these parameters were chosen forbetter visuals by trial & inspection
        sgm = 2*e

        # elipse equation
        ellipse = ((x_ang[:]/a)**2 + (y_ang[:]/b)**2)
        
        # fill the ellipse
        ellipse[ellipse<=1]=255
        ellipse[ellipse!=255]=0

        # apply exponential fading and gaussian blur
        ellipse = ellipse*np.exp(-(((x_ang[:,:]/(e*a))**2+(y_ang[:,:]/(e*b))**2))**0.5-sgm)
        ellipse = gaussian_filter(ellipse, sigma=sgm)

        # add ellipse channels to corresponding RGB image channels
        src[:,:,0] += ellipse*R
        src[:,:,1] += ellipse*G
        src[:,:,2] += ellipse*B


    return src/255




#======================================================================================
#|                            Coloured chessboard                                     |
#======================================================================================

# Create an image of coloured chessboard to observe shape distortions

def chessboard(size):
    '''
    This function generates a chessboard image. 4x4 squares. 
    
    Top Left - Red
    Top Right - Yellow
    Bottom Left - Green
    Bottom Right - Blue

    ----------------------------------------------------------------------
    Input:
    
    1) size - size of side of the image in pixels


    ----------------------------------------------------------------------
    Output:

    None

    Instead, saves the image as a .jpg file to the same directory as the calling file

    '''
    # source plane
    canvas = np.zeros(shape=(size,size,3))

    side = len(canvas[0,:])

    # Create figure and axes
    fig, ax = plt.subplots(layout="constrained")

    plt.tight_layout()

    # Display the image
    ax.set_axis_off()
    ax.imshow(canvas)
    # Create rectangle patches
    rect11 = patches.Rectangle((0, 0), side/4, side/4, linewidth=6, edgecolor='k', facecolor='r')
    rect21 = patches.Rectangle((0+side/4, 0), side/4, side/4, linewidth=6, edgecolor='k', facecolor='r')
    rect31 = patches.Rectangle((0+side/2, 0), side/4, side/4, linewidth=6, edgecolor='k', facecolor='y')
    rect41 = patches.Rectangle((0+3*side/4, 0), side/4, side/4, linewidth=6, edgecolor='k', facecolor='y')

    rect12 = patches.Rectangle((0, 0+side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='r')
    rect22 = patches.Rectangle((0+side/4, 0+side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='r')
    rect32 = patches.Rectangle((0+side/2, 0+side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='y')
    rect42 = patches.Rectangle((0+3*side/4, 0+side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='y')

    rect13 = patches.Rectangle((0, 0+side/2), side/4, side/4, linewidth=6, edgecolor='k', facecolor='g')
    rect23 = patches.Rectangle((0+side/4, 0+side/2), side/4, side/4, linewidth=6, edgecolor='k', facecolor='g')
    rect33 = patches.Rectangle((0+side/2, 0+side/2), side/4, side/4, linewidth=6, edgecolor='k', facecolor='b')
    rect43 = patches.Rectangle((0+3*side/4, 0+side/2), side/4, side/4, linewidth=6, edgecolor='k', facecolor='b')

    rect14 = patches.Rectangle((0, 0+3*side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='g')
    rect24 = patches.Rectangle((0+side/4, 0+3*side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='g')
    rect34 = patches.Rectangle((0+side/2, 0+3*side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='b')
    rect44 = patches.Rectangle((0+3*side/4, 0+3*side/4), side/4, side/4, linewidth=6, edgecolor='k', facecolor='b')

    # Add the patch to the Axes
    ax.add_patch(rect11)
    ax.add_patch(rect21)
    ax.add_patch(rect31)
    ax.add_patch(rect41)

    ax.add_patch(rect12)
    ax.add_patch(rect22)
    ax.add_patch(rect32)
    ax.add_patch(rect42)

    ax.add_patch(rect13)
    ax.add_patch(rect23)
    ax.add_patch(rect33)
    ax.add_patch(rect43)

    ax.add_patch(rect14)
    ax.add_patch(rect24)
    ax.add_patch(rect34)
    ax.add_patch(rect44)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    
    # save the figure
    fig.savefig('chssbrd.jpg', bbox_inches='tight',transparent=True, pad_inches=0)





#======================================================================================
#|                       Shape Distortion alignment map                               |
#======================================================================================

# Map shape distortions for lens with specified parameters



def alignment_map(size, rc, eps, rng, pbar=None):
    '''
    This function generates a map of shape distortions. 
    It visualizes for which allignment with the lens, the point source will be distorted the most
    distorted by comparing area of distorted image to the total area of image plane

    ----------------------------------------------------------------------
    Input:
    
    1) size - length of the source plane side in pixels (source has to be square)
    2) rc - core radius, from 0 to 1
    3) eps - ellipticity of the lens
    4) rng - range of values for reduced coordinate plane. 1 corresponds to (-1 -> 1)
    5) pbar - display progress bar when running the code. Increases computation time a bit

    ----------------------------------------------------------------------
    Output:

    1) Image object with map ofshape distortions

    '''

    # create empty arrays to host source image and distorted image
    src = np.zeros((size, size, 3))
    dist_vals = np.zeros((size, size))

    if pbar=="Y":
        range_variable_i = tqdm.tqdm(range(size), desc="Columns done")
        range_variable_j = tqdm.tqdm(range(size), leave=False)
    else:
        range_variable_i = range(size)
        range_variable_j = range(size)


    for i in range_variable_i:
        for j in range_variable_j:
            # set one and only one pixel to zero over each iteration, lens it
            src[i][j][:] = 1.0

            lensed_vals = lens(src, rc, eps, 3, rng)

            # count amount of bright pixel in the result
            pxls = np.count_nonzero((lensed_vals).all(axis = 2))

            # set ratio of bright pixels to total ratio to corresponding slot
            # in distortion image
            dist_vals[i][j] = pxls/(size*size)

            # set initial plane to zero before assigning next pixel
            src = src*0


    # increase the aluse to make them more clearly visible
    # with LogNorm applied
    dist_vals = dist_vals * 100

    # return the map
    return dist_vals







#======================================================================================
#|                                Caustics patterns                                   |
#======================================================================================




def caustic_pattern(size, lens_ellipticity, rc, rng, pbar="Y"):
    '''
    This function generates an image with inner caustic region (marked RED) and
    outer caustic region (GREEN).
    Does so by determining pixels which result in two or more shapes
    generated as result of lensing.

    ----------------------------------------------------------------------
    Input:
    
    1) source - Square image, RGB, pixel valuse 0 -> 255
    2) lens_ellipticity - eccentricity of the lens
    3) rc - core radius in reduced coordinates
    4) rng - range of reduced r. Default is set to 1 as in the validation test,
             so that reduced plane goes from -1 to 1


    ----------------------------------------------------------------------
    Output:

    1) Image object with inner and outer caustic regions marked

    '''
    # source has to be of uint8 dtype for OpenCV to be able to interpret it
    # and count contours
    src = np.zeros(shape=(size, size, 1)).astype('uint8')

    # empty array to host the patterns
    caustic_pattern = np.zeros(shape=(size, size, 3))

    if pbar=="Y":
        range_variable_i = tqdm.tqdm(range(size), desc="Columns done")
        range_variable_j = tqdm.tqdm(range(size), leave=False)
    else:
        range_variable_i = range(size)
        range_variable_j = range(size)


    for i in range_variable_i:
        for j in range_variable_j:

            # set one pixel to 1
            src[i][j][:] = 1.0

            # lens it
            lnsd = (lens(src, rc, lens_ellipticity, chnls = 1, rng = rng)).astype('uint8')

            # find contours of distorted shapes
            contours, hierarchy = cv2.findContours(lnsd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


            # count contours and mark the caustic regions
            if len(contours)>=2 and len(contours)<4:
                caustic_pattern[i][j][1]=1

            if len(contours)>=4:
                caustic_pattern[i][j][1]=0
                caustic_pattern[i][j][0]=1

            # set source plane to 0 before assigning next pixel
            src = src*0

    return caustic_pattern





#======================================================================================
#|                                Caustics contours                                   |
#======================================================================================






def caustic_contours(patterns):
    '''
    This function generates an image contour outline of the two caustics.
    Does so by filling the empty pixels within the region and generating a 
    contour using OpenCV.

    !!! Only works for elliptical/circular shape of outer caustic region
    due to implications of method used.

    ----------------------------------------------------------------------
    Input:
    
    1) patterns - patterns object generated by .caustic_pattern() function


    ----------------------------------------------------------------------
    Output:

    1) Image object with contour outline of inner and outer caustic

    '''
    # get size from the patterns image
    size = len(patterns[1,:,0])

    # set green channel to outer caustic
    # red channel to inner one
    inner = patterns[:,:,0]
    outer = patterns[:,:,1]

    # planes to contain the caustics before converting
    # them to uint8 dtype
    test_img_inner = np.zeros(shape = (size, size))
    test_img_outer = np.zeros(shape = (size, size))

    for i in range(size):

        # for each row, find nonzero pixels closest to the edge
        # fill the space between them
        if len(np.argwhere(inner[i,:]!=0))>0:

            left_inner = np.min(np.argwhere(inner[i,:]!=0))
            right_inner = np.max(np.argwhere(inner[i,:]!=0))

            inner[i,:][left_inner:right_inner]=1.0

            test_img_inner[i,:] += inner[i,:]

        # repeat the same but iterating over columns
        if len(np.argwhere(inner[:,i]!=0))>0:

            bot_inner = np.min(np.argwhere(inner[:,i]!=0))
            top_inner = np.max(np.argwhere(inner[:,i]!=0))

            inner[:,i][bot_inner:top_inner]=1.0

            test_img_inner[:,i] += inner[:,i]

        # same process but now for outer caustic
        if len(np.argwhere(outer[i,:]!=0))>0:

            left_outer = np.min(np.argwhere(outer[i,:]!=0))
            right_outer = np.max(np.argwhere(outer[i,:]!=0))

            outer[i,:][left_outer:right_outer]=1.0

            test_img_outer[i,:] += outer[i,:]

        if len(np.argwhere(outer[:,i]!=0))>0:

            bot_outer = np.min(np.argwhere(outer[:,i]!=0))
            top_outer = np.max(np.argwhere(outer[:,i]!=0))

            outer[:,i][bot_outer:top_outer]=1.0

            test_img_outer[:,i] += outer[:,i]


    # convert the filled in caustics to uint8
    inner_caustic_img = test_img_inner.astype('uint8')
    outer_caustic_img = test_img_outer.astype('uint8')

    # find contours of the regions
    ic_contour, hierarchy_ic = cv2.findContours(inner_caustic_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    inner_caustic_cnt = np.zeros(shape=(size, size, 3))
    cntrs_in = cv2.drawContours(inner_caustic_cnt, ic_contour, -1, (1, 0, 0), 1)

    oc_contour, hierarchy_ic = cv2.findContours(outer_caustic_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    outer_caustic_cnt = np.zeros(shape=(size, size, 3))
    cntrs_out = cv2.drawContours(outer_caustic_cnt, oc_contour, -1, (1, 0, 0), 1)

    # add outer caustic contour and inner contour to the one image plane
    cntrs_in += cntrs_out

    # return the contour image.
    return cntrs_in














