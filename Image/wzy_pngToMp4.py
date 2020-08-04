#https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
import cv2
import os

image_folder = '/tmp/difference_between_images/'
video_name = '1video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

# the second number n means speed x n
video = cv2.VideoWriter(video_name, 0, 10, (width , height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
# the output folder is the same folder of this python script