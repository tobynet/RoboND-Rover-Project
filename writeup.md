## Project: Search and Sample Return

[![Rover simulator Video](https://img.youtube.com/vi/bL0hAbSR-Oo/0.jpg)](https://youtu.be/bL0hAbSR-Oo)


📄 Click on image to see the simulator video.

---

### **The goals / steps of this project are the following:**

#### **Training / Calibration**

* ✅ Download the simulator and take data in "Training Mode"
* ✅ Test out the functions in the Jupyter Notebook provided
* ✅ Add functions to detect obstacles and samples of interest (golden rocks)
* ✅ Fill in the `process_image()` function with the appropriate image processing steps (perspective transform, color threshold etc.) to get from raw images to a map.  The `output_image` you create in this step should demonstrate that your mapping pipeline works.
* ✅ Use `moviepy` to process the images in your saved dataset with the `process_image()` function.  Include the video you produce as part of your submission.

#### **Autonomous Navigation / Mapping**

* ✅ Fill in the `perception_step()` function within the `perception.py` script with the appropriate image processing functions to create a map and update `Rover()` data (similar to what you did with `process_image()` in the notebook). 
* ✅ Fill in the `decision_step()` function within the `decision.py` script with conditional statements that take into consideration the outputs of the `perception_step()` in deciding how to issue throttle, brake and steering commands. 
* ✅ Iterate on your perception and decision function until your rover does a reasonable (need to define metric) job of navigating and mapping.

[//]: # (Image References)

[image1]: ./misc/rover_image.jpg
[image2]: ./calibration_images/example_grid1.jpg
[image3]: ./calibration_images/example_rock1.jpg
[simulator-settings04]: ./misc/simulator-settings04.png
[masked top image]: ./misc/masked-top-image00.jpg
[masked top image2]: ./misc/masked-top-image01.jpg
[vision video]: ./output/vision.mp4
[reduced terrain]: ./misc/reduce_navigable_terrain.png
[threshed only]: ./misc/threshed_only_image.jpg
[threshed only video]: ./output/threshed_only.mp4
[visualization rover trails]: ./misc/plot04.png
[world map]: ./misc/worldmap04.jpg
[modified world map1]: ./misc/worldmap04-modified01.png
[modified world map2]: ./misc/worldmap04-modified02.png
[world map video]: ./output/worldmap.mp4
[vision gif]: ./misc/vision.gif
[worldmap gif]: ./misc/worldmap.gif
[threshold gif]: ./misc/threshed_only.gif
[ex obstacles threshed only]: ./misc/obstacles-threshed_only.jpg
[ex obstacles vision]: ./misc/obstacles-vision.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/916/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf

You're reading it!

### Notebook Analysis

#### 1. Run the functions provided in the notebook on test images (first with the test data provided, next on data you have recorded). Add/modify functions to allow for color selection of obstacles and rock samples.

Here is an example of how to include an image in your writeup.

![alt text][image1]

#### 1. Populate the `process_image()` function with the appropriate analysis steps to map pixels identifying navigable terrain, obstacles and rock samples into a worldmap.  Run `process_image()` on your test data using the `moviepy` functions provided to create video output of your result.

And another!

![alt text][image2]

### Autonomous Navigation and Mapping

#### 1. Fill in the `perception_step()` (at the bottom of the `perception.py` script) and `decision_step()` (in `decision.py`) functions in the autonomous mapping scripts and an explanation is provided in the writeup of how and why these functions were modified as they were.

* In perception step:
  * **Mask top of camera images to reduce noise.**

    ![masked image][masked top image]
    ![masked image2][masked top image2]

    (🎥 [Here is the video link][vision video].)

    * To **ignore objects far from Rover camera** as inaccurate or noisy images.
    * Differ each image, `navigable terrain`, `rock_sample`, `obstacle mask`.

  * Using **HSV color space** image **to detect rocks accurately**.

    (At `find_rocks_by_hsv()`, `color_in_range_by_hsv()`)

  * **Reduce `navigable terrain image`** to improve fidelity. (Bottom left the image.)

    ![reduced terrain][reduced terrain]

    * Using `cv2.erode()` in `OpenCV`.
  * 🗺️ Modify mapping rocks process on the world map.
    * To ignore the rock found in obstacles.
    * Add helper flag and status(`Rover.found_rock`, `Rover.rock_pos`)
  * 🔎 Add some debugging code.
    * Generate threshold image to record video.
    (`Rover.threshed_only_image`)

    ![threshold image][threshed only]

    (🎥 [Here is the video link][threshed only video].)

* In decision step:
  * **Add `back` mode** to prevent getting stuck the rover.
  * **Add `approach rock` mode** to pick up rocks.
  * 🔎 Add more logging code to debug, visualization. (in `find_nearest_rock()`)

    ![visualization rover trails][visualization rover trails]

    > 📄 Above image is visualized from the log. ( [The notebook is here](./code/visualize_logs.ipynb).)

  * Add some simple and effective code **to estimate rock positions**.
    * To **check angle** against the rover : `is_accepted_angle_of_rocks()`
    * To **check distance** against the rover : `is_accepted_distance_of_rocks()`
    * To **check the attitude of the rover** : `is_accepted_attitude()`

      > 📝 To **prevent incorrect images from camera**.

  * Add sub-mode(`SubMode`) to easy to write nested mode, the decision tree.
  * **Add fluctuation in decision parameters** to prevent getting stuck, due to repeat the same action.
    * 💡 Simple implementation, using 🎲 `random.randint()`.

* Others:
  * 🔎 More visualize in the world map to easy to debug.

    * Add the **rover position and the angle** of yaw.

        ![modified world map][modified world map2]

    * Add the text main-mode and sub-mode of the rover.

        ![modified world map][modified world map1]

    (🎥 [Here is the video link][world map video].)

  * 🎥 Generate some videos to easy to debug.
    * [Threshold Only][threshed only video]

      ![vision gif][threshold gif]

    * [The Vision][vision video]

      ![vision gif][vision gif]

    * [The World map][world map video]

      ![worldmap][worldmap gif]

#### 2. Launching in autonomous mode your rover can navigate and map autonomously.  Explain your results and how you might improve them in your writeup.

> **Note: running the simulator with different choices of resolution and graphics quality may produce different results, particularly on different machines!  Make a note of your simulator settings (resolution and graphics quality set on launch) and frames per second (FPS output to terminal by `drive_rover.py`) in your writeup when you submit the project so your reviewer can reproduce your results.**

* Result:

  ```text
  Time: 906.3s
  Mapped: 95.8%
  Fidelity: 79.7%
  Rocks
    Located: 2
    Collected: 1
  ```

  * I thought it's easy to random searching only more than pick up samples, and return home.
  * And hard to **estimate the accurate position of rocks** only camera image.
    * I may be able to find more rocks(the number of `Located`) if I accept the inaccurate position of it.
    * But then, my rover will forward to nonexistence rocks, will be stuck.

* Improvement ideas:
  * **Implementation of the researched map** to prevent revisiting and comfortable pathfinding, return to home.
  * **Implementation of avoids obstacles in open space** using elaborate image processing.

    Ex.

      ![vision][ex obstacles vision]

  * **Improvement object recognition accuracy** to comfort pathfinding.
  * Improvement to estimate the accurate position of rocks: Humm..?
    * Probabilistic localization?

* 📄 Simulator settings:

  ![simulator settings][simulator-settings04]

  * Screen resolution: `640 x 480` (or `800 x 600`)
  * Graphics quality: `Good`
  * FPS: `20`
  * Screen Recorder: `Game DVR on Windows 10`
    * Press `Win + Alt + R` to record window.
