import numpy as np
import cv2
from copy import copy
from scipy.signal import convolve2d


def load_video(video_path, save_face=False, roi_dim = (96,96) , only_face_frames = False):

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save_face:
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    full_video = []
    face_frames = []

    ret = True
    while ret:

        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # frame = frame_bgr.copy()
        # temp = None
        # temp = frame[:,:,0]
        # frame[:,:,0] = frame[:,:,2]
        # frame[:,:,2] = temp

        if save_face:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.05, 7)

            if len(faces)==1:
                for (x, y, w, h) in faces:

                    #roi_color = frame[y:y + h, x:x + w]
                    roi_color = cv2.resize(frame[y:y + h, x:x + w], roi_dim, interpolation =cv2.INTER_CUBIC)
                    face_frames.append(roi_color)

        full_video.append(frame)

    print('Video loaded')

    if only_face_frames:
        return np.array(face_frames) , fps
    else:
        if not save_face:
            return np.array(full_video) , fps
    if save_face:
        return np.array(full_video) , np.array(face_frames) , fps


def video_resize(video, factor):
    return video[:,::factor,::factor,:]


def get_kernel(dim=None, sigma=None):

    if dim is not None and sigma is not None:
        kernel = np.zeros((dim, dim))
        for y in range(dim):
            for x in range(dim):
                kernel[x][y] = np.exp(-(abs(y-int(dim/2))**2+abs(xint(dim/2))**2)/(2*sigma**2))/(2*np.pi*sigma**2)
        return kernel
    else:
        # Default kernel
        kernel = 1/273 * np.array([[1,4,7,4,1] ,
                                    [4,16,26,16,4] ,
                                    [7,26,41,26,7] ,
                                    [4,16,26,16,4] ,
                                    [1,4,7,4,1]])
        return kernel

def gaussian_pyramid(image, levels, kernel=None):

    if kernel == None:
        kernel = get_kernel()

    gauss = image.copy()
    gauss_pyr = [gauss]

    for level in range(1, levels):
        for channel in range(3):
            gauss[:,:,channel] = convolve2d(gauss[:,:,channel], kernel, 'same')

        gauss = gauss[::2,::2,:]
        gauss_pyr.append(gauss)

    return gauss_pyr


def gaussian_video(video, levels, kernel=None):

    levels_vid_data = []
    for level in range(1, levels):
        for frame_nr in range(0, len(video)):
            frame = video[frame_nr]
            pyr = gaussian_pyramid(frame, levels, kernel)
            gaussian_frame = pyr[level] # use only highest gaussian level is used
            if frame_nr == 0: # initialize for the first time
                vid_gauss_data = np.zeros((len(video), gaussian_frame.shape[0], gaussian_frame.shape[1], 3))

            vid_gauss_data[frame_nr] = gaussian_frame

        levels_vid_data.append(vid_gauss_data)

    return levels_vid_data


def rgb2yiq(video):

    '''
    Converts the video color from RGB to YIQ (NTSC)
    '''

    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.322],
                             [0.211, -0.523, 0.312]])

    t = np.dot(video, yiq_from_rgb.T)

    return t


def yiq2rgb(video):

    "Converts the video color from YIQ (NTSC) to RGB"
    rgb_from_yiq = np.array([[1, 0.956, 0.621],
                             [1, -0.272, -0.647],
                             [1, -1.106, 1.703]])
    t = np.dot(video, rgb_from_yiq.T)
    return t


def bgr2rgb(frame):

    return frame[:, :, [2, 1, 0]]


if __name__ == "__main__":

    # test for gaussian pyramid
    img = cv2.imread(r'C:\Users\user\Desktop\Facultate\test\1_sal.png', cv2.IMREAD_COLOR)
    pyr = gaussian_pyramid(img, levels=5)