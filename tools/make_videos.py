import os.path
import sys
from glob import glob
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip, ImageSequenceClip

OUTPUT_DIR = 'output'

# Create mp4 from jpegs
def save_with_mp4(output_dir, name):
    glob_name = os.path.join(output_dir, name, '*.jpg')
    images = glob(glob_name)
    if len(images) == 0:
        print('Image not found in {}'.format(glob_name), file=sys.stderr)
        return
    clip = ImageSequenceClip(images, fps=60)
    clip.write_videofile(os.path.join(output_dir, '{}.mp4'.format(name)))

save_with_mp4(OUTPUT_DIR, 'rover')
save_with_mp4(OUTPUT_DIR, 'threshed_only')
save_with_mp4(OUTPUT_DIR, 'vision')
save_with_mp4(OUTPUT_DIR, 'worldmap')