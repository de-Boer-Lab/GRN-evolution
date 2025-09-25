import os
import moviepy.video.io.ImageSequenceClip
from natsort import natsorted, ns
from PIL import Image

image_folder='for-submission/combined_grid/'
fps=40

image_files = [os.path.join(image_folder,img)
               for img in natsorted(os.listdir(image_folder), alg=ns.IGNORECASE)
               if img.endswith(".png")]

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('for-submission/evolution.mp4')