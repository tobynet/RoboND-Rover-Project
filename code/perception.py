import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh=(160, 160, 160)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                & (img[:,:,1] > rgb_thresh[1]) \
                & (img[:,:,2] > rgb_thresh[2])
    # Index the array of zeros with the boolean array and set to 1
    color_select[above_thresh] = 1
    # Return the binary image
    return color_select

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

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image
    
    return warped

# color in range(HSV version)
def color_in_range_by_hsv(bgr_img, hsv_min, hsv_max):
    # ref. https://docs.opencv.org/3.1.0/d7/d1b/group__imgproc__misc.html#ga397ae87e1288a81d2363b61574eb8cab
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)  # rgb to hsv
    
    # 0 or 255 before filtering
    # ref. https://docs.opencv.org/3.1.0/d2/de8/group__core__array.html#ga48af0ab51e36436c5d04340e036ce981
    return cv2.inRange(hsv_img, 
        lowerb=np.array(hsv_min, np.uint8), 
        upperb=np.array(hsv_max, np.uint8))

## Find rock (岩を探す)
def find_rocks_by_hsv(bgr_img, hsv_min = (60, 120, 110), hsv_max = (135, 255, 255)):
    # return 0 or 255 image
    return color_in_range_by_hsv(bgr_img, hsv_min, hsv_max)

# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()

    # NOTE: camera image is coming to you in Rover.img
    # 1) Define source and destination points for perspective transform

    img = Rover.img

    dst_size = 5 
    bottom_offset = 6
    source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    destination = np.float32([[img.shape[1]/2 - dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - bottom_offset],
                  [img.shape[1]/2 + dst_size, img.shape[0] - 2*dst_size - bottom_offset], 
                  [img.shape[1]/2 - dst_size, img.shape[0] - 2*dst_size - bottom_offset],
                  ])
    scale = 10
    world_size = Rover.worldmap.shape[0]

    xpos, ypos, yaw = (Rover.pos[0], Rover.pos[1], Rover.yaw)

    # 2) Apply perspective transform
    _warped_img = perspect_transform(img, source, destination)
    warped_obst_img = perspect_transform(np.ones_like(img[:,:,0]), source, destination)

    # Far distance consider as obstacles (sky or...)
    masked_height = int(_warped_img.shape[0] / 4)
    #h, w, _ = img.shape
    warped_img = np.copy(_warped_img)
    warped_img[:100,:] = 0
    #warped_img[:60,:] = 0
    #warped_img[:,:int(w/6)] = 0
    #warped_img[:,int(w/6):] = 0
    #warped_img[:80,:] = 0
    warped_obst_img[:80,:] = 0

    # 3) Apply color threshold to identify navigable terrain/obstacles/rock samples
    _threshed_navi_img = color_thresh(warped_img, (160, 160, 160))
    threshed_obst_img = np.absolute(np.float32(_threshed_navi_img) - 1) * warped_obst_img

    # # Reduce image to improve fidelity
    # # ref. https://docs.opencv.org/3.1.0/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5,5), np.uint8)
    # kernel = np.array([
    #     [0,1,1,1,0],
    #     [0,1,1,1,0],
    #     [0,1,1,1,0],
    #     [0,0,0,0,0],
    #     [0,0,0,0,0]], np.uint8) 
    threshed_navi_img = cv2.erode(_threshed_navi_img, kernel)
    #threshed_navi_img = _threshed_navi_img


    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = navigable terrain color-thresholded binary image
    # navigatable terrain as BLUE on rover view
    Rover.vision_image[:,:,2] = threshed_navi_img * 255
    # obstacles as RED on rover view
    Rover.vision_image[:,:,0] = threshed_obst_img * 255

    # 5) Convert map image pixel values to rover-centric coords
    navi_xpix, navi_ypix = rover_coords(threshed_navi_img)
    obst_xpix, obst_ypix = rover_coords(threshed_obst_img)

    # 6) Convert rover-centric pixel values to world coordinates
    obst_x_world, obst_y_world = pix_to_world(
        obst_xpix, obst_ypix, xpos, ypos, yaw, world_size, scale)
    navi_x_world, navi_y_world = pix_to_world(
        navi_xpix, navi_ypix, xpos, ypos, yaw, world_size, scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1

    # navigatable terrain as BLUE on worldmap
    #Rover.worldmap[navi_y_world, navi_y_world, 2] += 30
    #Rover.worldmap[navi_y_world, navi_y_world, 2] += 100
    #Rover.worldmap[navi_y_world, navi_x_world] = (0, 0, 255)
    Rover.worldmap[navi_y_world, navi_x_world, 2] = 255
    # obstacles as RED on worldmap
    Rover.worldmap[obst_y_world, obst_x_world, 0] += 1
    #Rover.worldmap[obst_y_world, obst_x_world] = (255, 0, 0)
    #Rover.worldmap[obst_y_world, obst_x_world, 0] = 255

    # override blue on red
    #Rover.worldmap[Rover.worldmap[:,:,2] > 0, 0] = 0

    # 8) Convert rover-centric pixel positions to polar coordinates
    dist, angles = to_polar_coords(navi_xpix, navi_ypix)
    # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles
    
    Rover.nav_dists = dist
    Rover.nav_angles = angles

    # find rocks
    #rocks_img = find_rocks_by_hsv(_warped_img)
    #_img = np.copy(_warped_img)
    rocks_img = find_rocks_by_hsv(_warped_img) # Ignore rocks too far
    rocks_img[:40,:] = 0

    if rocks_img.any():
        # rocks as GREEN on rover view
        Rover.vision_image[:, :, 1] = rocks_img * 255
        #Rover.vision_image[rocks_img > 0, 1] = 255 
        #Rover.vision_image[rocks_img > 0] = (255, 255, 255)  # highlight

        rock_xpix, rock_ypix = rover_coords(rocks_img)
        rock_x_world, rock_y_world = pix_to_world(
            rock_xpix, rock_ypix, xpos, ypos, yaw, world_size, scale)
        
        # Get a point that most closely rocks
        rock_dist, rock_angles = to_polar_coords(rock_xpix, rock_ypix)
        
        rock_i = np.argmin(rock_dist)   # Select nearest rock
        #rock_i = np.argsort(rock_dist)[len(rock_dist)//2]  # Select middle rock

        rock_world_x = rock_x_world[rock_i]
        rock_world_y = rock_y_world[rock_i]

        # Ignore the rock in obstacles!
        if Rover.worldmap[rock_world_x, rock_world_x, 0] > 0:
        #if Rover.worldmap[rock_world_x, rock_world_x, 2] > 0:
            # Set flag and data for decision
            Rover.found_rock = True
            Rover.rock_pos = (rock_world_x, rock_world_y)

            # Mark with a dot of rock
            #   see `create_output_images()` in `supporting_functions.py`
            Rover.worldmap[rock_world_y, rock_world_x, 1] = 255
        else:
            Rover.found_rock = False
            Rover.rock_pos = None
    else:
        Rover.vision_image[:, :, 1] = 0
        Rover.found_rock = False


    # DEBUG: Make threshed only image before transform
    _thres_img = color_thresh(img)
    Rover.threshed_only_image[:,:,2] = _thres_img * 255
    Rover.threshed_only_image[:,:,0] = np.absolute(np.float32(_thres_img) - 1) * np.ones_like(img[:,:,0])
    Rover.threshed_only_image[:,:,1] = find_rocks_by_hsv(img)

    return Rover