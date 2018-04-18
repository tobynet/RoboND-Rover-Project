
# coding: utf-8

# ## Rover Project Test Notebook
# This notebook contains the functions from the lesson and provides the scaffolding you need to test out your mapping methods.  The steps you need to complete in this notebook for the project are the following:
# 
# * First just run each of the cells in the notebook, examine the code and the results of each.
# * Run the simulator in "Training Mode" and record some data. Note: the simulator may crash if you try to record a large (longer than a few minutes) dataset, but you don't need a ton of data, just some example images to work with.   
# * Change the data directory path (2 cells below) to be the directory where you saved data
# * Test out the functions provided on your data
# * Write new functions (or modify existing ones) to report and map out detections of obstacles and rock samples (yellow rocks)
# * Populate the `process_image()` function with the appropriate steps/functions to go from a raw image to a worldmap.
# * Run the cell that calls `process_image()` using `moviepy` functions to create video output
# * Once you have mapping working, move on to modifying `perception.py` and `decision.py` to allow your rover to navigate and map in autonomous mode!
# 
# **Note: If, at any point, you encounter frozen display windows or other confounding issues, you can always start again with a clean slate by going to the "Kernel" menu above and selecting "Restart & Clear Output".**
# 
# **Run the next cell to get code highlighting in the markdown cells.**

# In[467]:


get_ipython().run_cell_magic('HTML', '', '<style> code {background-color : orange !important;} </style>')


# In[468]:


get_ipython().run_line_magic('matplotlib', 'inline')
#%matplotlib qt # Choose %matplotlib qt to plot to an interactive window (note it may show up behind your browser)
# Make some of the relevant imports
import cv2 # OpenCV for perspective transform
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc # For saving images as needed
import glob  # For reading in a list of images from a folder
import imageio
imageio.plugins.ffmpeg.download()


# ## Quick Look at the Data
# There's some example data provided in the `test_dataset` folder.  This basic dataset is enough to get you up and running but if you want to hone your methods more carefully you should record some data of your own to sample various scenarios in the simulator.  
# 
# Next, read in and display a random image from the `test_dataset` folder

# In[469]:


path = '../test_dataset/IMG/*'
img_list = glob.glob(path)
# Grab a random image and display it
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
plt.imshow(image)


# ## Calibration Data
# Read in and display example grid and rock sample calibration images.  You'll use the grid for perspective transform and the rock image for creating a new color selection that identifies these samples of interest. 

# In[470]:


# In the simulator you can toggle on a grid on the ground for calibration
# You can also toggle on the rock samples with the 0 (zero) key.  
# Here's an example of the grid and one of the rocks
example_grid = '../calibration_images/example_grid1.jpg'
example_rock = '../calibration_images/example_rock1.jpg'
grid_img = mpimg.imread(example_grid)
rock_img = mpimg.imread(example_rock)

fig = plt.figure(figsize=(12,3))
plt.subplot(121)
plt.imshow(grid_img)
plt.subplot(122)
plt.imshow(rock_img)


# ## Perspective Transform
# 
# Define the perspective transform function from the lesson and test it on an image.

# In[471]:


# Define a function to perform a perspective transform
# I've used the example grid image above to choose source points for the
# grid cell in front of the rover (each grid cell is 1 square meter in the sim)
# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped


# Define calibration box in source (actual) and destination (desired) coordinates
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter
# The destination box will be 2*dst_size on each side
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 6
source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
destination = np.float32([[image.shape[1]/2 - dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - bottom_offset],
                  [image.shape[1]/2 + dst_size, image.shape[0] - 2*dst_size - bottom_offset], 
                  [image.shape[1]/2 - dst_size, image.shape[0] - 2*dst_size - bottom_offset],
                  ])
warped = perspect_transform(grid_img, source, destination)
plt.imshow(warped)
#scipy.misc.imsave('../output/warped_example.jpg', warped)


# ## Color Thresholding
# Define the color thresholding function from the lesson and apply it to the warped image
# 
# **TODO:** Ultimately, you want your map to not just include navigable terrain but also obstacles and the positions of the rock samples you're searching for.  Modify this function or write a new function that returns the pixel locations of obstacles (areas below the threshold) and rock samples (yellow rocks in calibration images), such that you can map these areas into world coordinates as well.  
# **Hints and Suggestion:** 
# * For obstacles you can just invert your color selection that you used to detect ground pixels, i.e., if you've decided that everything above the threshold is navigable terrain, then everthing below the threshold must be an obstacle!
# 
# 
# * For rocks, think about imposing a lower and upper boundary in your color selection to be more specific about choosing colors.  You can investigate the colors of the rocks (the RGB pixel values) in an interactive matplotlib window to get a feel for the appropriate threshold range (keep in mind you may want different ranges for each of R, G and B!).  Feel free to get creative and even bring in functions from other libraries.  Here's an example of [color selection](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html) using OpenCV.  
# 
# * **Beware However:** if you start manipulating images with OpenCV, keep in mind that it defaults to `BGR` instead of `RGB` color space when reading/writing images, so things can get confusing.

# In[472]:


# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0])                 & (img[:,:,1] > rgb_thresh[1])                 & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

fig = plt.figure(figsize=(12,7))
fig.tight_layout()

threshed = color_thresh(warped)
plt.subplot(221); plt.imshow(threshed, cmap='gray')

#scipy.misc.imsave('../output/warped_threshed.jpg', threshed*255)
plt.show()


# In[473]:


# Identify pixels above the range using hsv color space 
def color_in_range_by_hsv(bgr_img, hsv_min, hsv_max):
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)  # rgb to hsv
    lower = np.array(hsv_min, np.uint8)   # to numpy array
    upper = np.array(hsv_max, np.uint8)   # to numpy array
    return cv2.inRange(hsv_img, lower, upper)  # filter

def rgb_to_hsv_color(rgb_color):
    hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]

    print('origin rgb -> hsv color(sv%):', rgb_color, hsv_color, (hsv_color[1]/255, hsv_color[2]/255))
    return hsv_color

# rock color in hsv
hsv_rock_color = rgb_to_hsv_color((150, 130, 18))



fig = plt.figure(figsize=(21,7))
fig.tight_layout()

# For rocks (yellow)
plt.subplot(231); plt.imshow(rock_img)

# color in range
hsv_rock_min = (60, 120, 110)
hsv_rock_max = (135, 255, 255)
threshed_hsb_rock_img = color_in_range_by_hsv(rock_img, hsv_rock_min, hsv_rock_max)

plt.subplot(232); plt.imshow(threshed_hsb_rock_img, cmap='gray')

# masked
masked_hsb_rock_img = cv2.bitwise_and(rock_img, rock_img, mask=threshed_hsb_rock_img)
plt.subplot(233); plt.imshow(masked_hsb_rock_img)  


# perspect
warped_with_rock = perspect_transform(rock_img, source, destination)
plt.subplot(234); plt.imshow(warped_with_rock)

# color in range
threshed_warped_rock_img = color_in_range_by_hsv(warped_with_rock, hsv_rock_min, hsv_rock_max)
plt.subplot(235); plt.imshow(threshed_warped_rock_img, cmap='gray')

# masked
masked_warped_rock_img = cv2.bitwise_and(warped_with_rock, warped_with_rock, mask=threshed_warped_rock_img)
plt.subplot(236); plt.imshow(masked_warped_rock_img)  


# In[474]:


## Find rock (岩を探す)
def color_in_range_by_hsv(bgr_img, hsv_min, hsv_max):
    # ref. https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)  # rgb to hsv
    
    # ref. https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    lower = np.array(hsv_min, np.uint8)   # to numpy array
    upper = np.array(hsv_max, np.uint8)   # to numpy array
    return cv2.inRange(hsv_img, lower, upper)  # 0 or 255 before filtering

def find_rocks_by_hsv(img, hsv_min = (60, 120, 110), hsv_max = (135, 255, 255)):
    # return 0 or 255 image
    return color_in_range_by_hsv(img, hsv_min, hsv_max)

threshed_rocks_img = find_rocks_by_hsv(rock_img)
not_rocks_img = find_rocks_by_hsv(image)
print('rocks:', threshed_rocks_img.any(), np.count_nonzero(threshed_rocks_img))
print('not rocks: ', not_rocks_img.any(), np.count_nonzero(not_rocks_img))

fig = plt.figure(figsize=(14,7))
plt.subplot(121); plt.imshow(rock_img)
plt.subplot(122); plt.imshow(threshed_rocks_img, cmap='gray')


# In[475]:


# Identify pixels above the range
def color_in_range_by_bgr(bgr_img, bgr_min, bgr_max):
    lower = np.array(bgr_min, np.uint8)   # to numpy array
    upper = np.array(bgr_max, np.uint8)   # to numpy array
    return cv2.inRange(bgr_img, lower, upper)  # filter


rgb_to_hsv_color((0, 0, 0)) # dark obstacle
rgb_to_hsv_color((73, 68, 65)) # light obstacle
rgb_to_hsv_color((114, 105, 99)) # sky
rgb_to_hsv_color((197, 178, 163)) # ground
rgb_to_hsv_color((189, 157, 0)) # rock


# In[476]:


## For obstacles only (mask) 障害のみ
# make fill image
full_img = np.ones_like(rock_img[:,:,0])
# transform image
warped_full_img = perspect_transform(full_img, source, destination)

fig = plt.figure(figsize=(14,7))
fig.tight_layout()

plt.subplot(221); plt.imshow(rock_img)
plt.subplot(222); plt.imshow(warped_with_rock)
plt.subplot(223); plt.imshow(full_img, cmap='gray')
plt.subplot(224); plt.imshow(warped_full_img, cmap='gray')  # 


# In[477]:


## For navigable terrain only 道のみ

#hsv_navi_min = (10, 30, 120)
#hsv_navi_max = (255, 230, 255)

fig = plt.figure(figsize=(21,7))
fig.tight_layout()

plt.subplot(131); plt.imshow(rock_img)
plt.subplot(132); plt.imshow(warped_with_rock)

normal_threshed_navi = color_thresh(warped_with_rock)
#_img = color_in_range_by_hsv(warped_with_rock, hsv_navi_min, hsv_navi_max)

plt.subplot(133); plt.imshow(normal_threshed_navi, cmap='gray')
threshed_navi = normal_threshed_navi


# ## Coordinate Transformations
# Define the functions used to do coordinate transforms and apply them to an image.

# In[478]:


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the 
    # center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel

# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Grab another random image
idx = np.random.randint(0, len(img_list)-1)
image = mpimg.imread(img_list[idx])
warped = perspect_transform(image, source, destination)
threshed = color_thresh(warped)

# Calculate pixel values in rover-centric coords and distance/angle to all pixels
xpix, ypix = rover_coords(threshed)
dist, angles = to_polar_coords(xpix, ypix)
mean_dir = np.mean(angles)

# Do some plotting
fig = plt.figure(figsize=(12,9))
plt.subplot(221)
plt.imshow(image)
plt.subplot(222)
plt.imshow(warped)
plt.subplot(223)
plt.imshow(threshed, cmap='gray')
plt.subplot(224)
plt.plot(xpix, ypix, '.')
plt.ylim(-160, 160)
plt.xlim(0, 160)
arrow_length = 100
x_arrow = arrow_length * np.cos(mean_dir)
y_arrow = arrow_length * np.sin(mean_dir)
plt.arrow(0, 0, x_arrow, y_arrow, color='red', zorder=2, head_width=10, width=2)


# ## Read in saved data and ground truth map of the world
# The next cell is all setup to read your saved data into a `pandas` dataframe.  Here you'll also read in a "ground truth" map of the world, where white pixels (pixel value = 1) represent navigable terrain.  
# 
# After that, we'll define a class to store telemetry data and pathnames to images.  When you instantiate this class (`data = Databucket()`) you'll have a global variable called `data` that you can refer to for telemetry and map data within the `process_image()` function in the following cell.  
# 

# In[479]:


# Import pandas and read in csv file as a dataframe
import pandas as pd
# Change the path below to your data directory
# If you are in a locale (e.g., Europe) that uses ',' as the decimal separator
# change the '.' to ','
df = pd.read_csv('../test_dataset/robot_log.csv', delimiter=';', decimal='.')
csv_img_list = df["Path"].tolist() # Create list of image pathnames
# Read in ground truth map and create a 3-channel image with it
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

# Creating a class to be the data container
# Will read in saved data from csv file and populate this object
# Worldmap is instantiated as 200 x 200 grids corresponding 
# to a 200m x 200m space (same size as the ground truth map: 200 x 200 pixels)
# This encompasses the full range of output position values in x and y from the sim
class Databucket():
    def __init__(self):
        self.images = csv_img_list  
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        self.count = 0 # This will be a running index
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
        self.ground_truth = ground_truth_3d # Ground truth worldmap

# Instantiate a Databucket().. this will be a global variable/object
# that you can refer to in the process_image() function below
data = Databucket()


# In[480]:


############# Test of make worldmap ################

# a camera image to navigable terrain/obstacles/rock samples
_worldmap = np.copy(data.worldmap)
# todo: update _worldmap
print(_worldmap.shape)

_index = 0
_img = mpimg.imread(data.images[_index])
_scale = 10
_xpos = data.xpos[_index]
_ypos = data.ypos[_index]
_yaw = data.yaw[_index]
print(_img.shape, _xpos, _ypos, _yaw)


# fill for obstacles terrain to 
_full_img = np.ones_like(_img[:,:,0])
_warped_full_img = perspect_transform(_full_img, source, destination)
_xpix, _ypix = rover_coords(_warped_full_img)
_x_world, _y_world = pix_to_world(_xpix, _ypix, _xpos, _ypos, _yaw, _worldmap.shape[0], _scale)
_worldmap[_y_world,_x_world,:] = (255,0,0)

# fill for navigable terrain to blue
_warped_img = perspect_transform(_img, source, destination)
_threshed_navi = color_thresh(_warped_img)
_xpix, _ypix = rover_coords(_threshed_navi)
_x_world, _y_world = pix_to_world(_xpix, _ypix, _xpos, _ypos, _yaw, _worldmap.shape[0], _scale)
_worldmap[_y_world,_x_world,:] = (0,0,255)

# fill for rock samples
_threshed_hsb_rock_img = color_in_range_by_hsv(_warped_img, hsv_rock_min, hsv_rock_max)
_xpix, _ypix = rover_coords(_threshed_hsb_rock_img)
_x_world, _y_world = pix_to_world(_xpix, _ypix, _xpos, _ypos, _yaw, _worldmap.shape[0], _scale)
_worldmap[_y_world,_x_world,:] = (255,255,255)


# overlay 
_map_add = cv2.addWeighted(_worldmap, 1, data.ground_truth, 0.5, 0)
_flippud_map_add = np.flipud(_map_add)

# show 
fig = plt.figure(figsize=(18,5))
print(np.max(_worldmap), np.max(data.ground_truth), np.max(_map_add))
plt.subplot(131); plt.imshow(np.uint8(_worldmap))
plt.subplot(132); plt.imshow(np.uint8(data.ground_truth))
plt.subplot(133); plt.imshow(np.uint8(_map_add))


# ## Write a function to process stored images
# 
# Modify the `process_image()` function below by adding in the perception step processes (functions defined above) to perform image analysis and mapping.  The following cell is all set up to use this `process_image()` function in conjunction with the `moviepy` video processing package to create a video from the images you saved taking data in the simulator.  
# 
# In short, you will be passing individual images into `process_image()` and building up an image called `output_image` that will be stored as one frame of video.  You can make a mosaic of the various steps of your analysis process and add text as you like (example provided below).  
# 
# 
# 
# To start with, you can simply run the next three cells to see what happens, but then go ahead and modify them such that the output video demonstrates your mapping process.  Feel free to get creative!

# In[481]:



# Define a function to pass stored images to
# reading rover position and yaw angle from csv file
# This function will be used by moviepy to create an output video
def process_image(img):
    # Example of how to use the Databucket() object defined above
    # to print the current x, y and yaw values 
    # print(data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])

    # DONE: 
    # 1) Define source and destination points for perspective transform
    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    
    scale = 10
    world_size = data.worldmap.shape[0]
    xpos, ypos, yaw = (data.xpos[data.count], data.ypos[data.count], data.yaw[data.count])
    
    # 2) Apply perspective transform
    warped_img = perspect_transform(img, source, destination)
    filled_img = np.ones_like(img[:,:,0])
    warped_obst_img = perspect_transform(filled_img, source, destination)
    
    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    threshed_navi_img = color_thresh(warped_img)
    # reserve mask only(obst_img - img)
    threshed_obst_img =         np.absolute(np.float32(threshed_navi_img) - 1) * warped_obst_img
    
    # 4) Convert thresholded image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(threshed_navi_img)
    obst_xpix, obst_ypix = rover_coords(threshed_obst_img)
    
    # 5) Convert rover-centric pixel values to world coords
    obst_x_world, obst_y_world = pix_to_world(
        obst_xpix, obst_ypix, xpos, ypos, yaw, world_size, scale)
    navi_x_world, navi_y_world = pix_to_world(
        navi_xpix, navi_ypix, xpos, ypos, yaw, world_size, scale)

    # 6) Update worldmap (to be displayed on right side of screen)
    # Example: data.worldmap[obst_y_world, obst_x_world, 0] += 1
    #          data.worldmap[rock_y_world, rock_x_world, 1] += 1
    #          data.worldmap[navi_y_world, navi_x_world, 2] += 1
    
    # navigatable terrain as BLUE
    data.worldmap[navi_y_world, navi_x_world, 2] = 255
    # obstacles as RED
    data.worldmap[obst_y_world, obst_x_world, 0] = 255
    # override blue on red
    data.worldmap[data.worldmap[:,:,2] > 0, 0] = 0
    
    # Fill by white if rocks in image
    rocks_img = find_rocks_by_hsv(warped_img)
    if rocks_img.any():
        rock_xpix, rock_ypix = rover_coords(rocks_img)    
        rock_x_world, rock_y_world = pix_to_world(
            rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
        # rocks as WHITE
        data.worldmap[rock_y_world, rock_x_world, :] = 255

    # 7) Make a mosaic image, below is some example code
    # First create a blank image (can be whatever shape you like)
    output_image = np.zeros((img.shape[0] + data.worldmap.shape[0], img.shape[1]*2, 3))
    
    # Next you can populate regions of the image with various output
    # Here I'm putting the original image in the upper left hand corner
    # 左上に元画像を貼り付ける
    output_image[0:img.shape[0], 0:img.shape[1]] = img

    # Let's create more images to add to the mosaic, first a warped image
    warped = perspect_transform(img, source, destination)
    # Add the warped image in the upper right hand corner
    # 右上に透視変換した画像を貼り付ける
    output_image[0:img.shape[0], img.shape[1]:] = warped

    # Overlay worldmap with ground truth map
    map_add = cv2.addWeighted(data.worldmap, 1, data.ground_truth, 0.5, 0)
    # Flip map overlay so y-axis points upward and add to output_image 
    # 左下に、 実際の地面と 3 つの identify 情報が合成されたマップを貼り付ける
    output_image[img.shape[0]:, 0:data.worldmap.shape[1]] = np.flipud(map_add)

    # World map only 
    # 右下に、 ワールドマップ だけのマップを貼り付ける
    output_image[img.shape[0]:, 
                 img.shape[1]:img.shape[1] + data.worldmap.shape[1]] \
        = np.flipud(data.worldmap)

    # Then putting some text over the image
    cv2.putText(output_image,"Populate this image with your analyses to make a video!", (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 255, 255), 1)
    if data.count < len(data.images) - 1:
        data.count += 1 # Keep track of the index in the Databucket()
    
    return output_image


# ## Make a video from processed image data
# Use the [moviepy](https://zulko.github.io/moviepy/) library to process images and create a video.
#   

# In[482]:


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip


# Define pathname to save the output video
output = '../output/test_mapping.mp4'
data = Databucket() # Re-initialize data in case you're running this cell multiple times
clip = ImageSequenceClip(data.images, fps=60) # Note: output video will be sped up because 
                                          # recording rate in simulator is fps=25
new_clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().run_line_magic('time', 'new_clip.write_videofile(output, audio=False)')


# ### This next cell should function as an inline video player
# If this fails to render the video, try running the following cell (alternative video rendering method).  You can also simply have a look at the saved mp4 in your `/output` folder

# In[392]:



from IPython.display import HTML
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(output))


# ### Below is an alternative way to create a video in case the above cell did not work.

# In[387]:


import io
import base64
video = io.open(output, 'r+b').read()
encoded_video = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded_video.decode('ascii')))

