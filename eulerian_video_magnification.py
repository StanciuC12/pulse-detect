from video_functions2 import *
import scipy.fftpack as fftpack
import numpy as np
import cv2


def video2evm(video_frames, alpha, lowcut_freq, highcut_freq, chrom_attenuation=None, fs=30, levels=4):

    " Returns colour magnified video after creating a fft from the whole movie"
    "(pixel by pixel), applying an ideal filter to select only the frequencies of inetrest"
    "and adding the last level gaussian pyramid element"

    # from rgb to yiq
    video_frames = rgb2yiq(video_frames) #luminance (Y) and chrominance (I and Q)
    # build Gaussian pyramid and use the highest level
    high_gauss_vid = gaussian_video(video_frames, levels)
    for level_vid in high_gauss_vid:

        # apply fft
        fft = fftpack.rfft(level_vid, axis=0)
        frequencies = fftpack.rfftfreq(fft.shape[0], d=1.0 / fs) # sample frequencies
        bp_filter = np.logical_and(frequencies > lowcut_freq, frequencies < highcut_freq) # logical array if values between low and high frequencies
        fft[~bp_filter] = 0 # cutoff values outside the bandpass
        filtered = fftpack.irfft(fft, axis=0) # inverse fourier transformation
        filtered *= alpha # magnification

        # chromatic attenuation
        if chrom_attenuation is not None:

            filtered[:, :, :, 1] = filtered[:, :, :, 1] * chrom_attenuation
            filtered[:, :, :, 2] = filtered[:, :, :, 2] * chrom_attenuation

        # resize last gaussian level to the frames size
        filtered_video_list = np.zeros(video_frames.shape)
        for i in range(len(video_frames)):
            f = filtered[i]
            filtered_video_list[i] = cv2.resize(f, (video_frames.shape[2], video_frames.shape[1]))

        final = filtered_video_list/len(high_gauss_vid)
        # Add to original
        video_frames = final + video_frames

    # from yiq to rgb
    video_frames = yiq2rgb(video_frames)
    # Cutoff wrong values
    video_frames[video_frames < 0] = 0
    video_frames[video_frames > 255] = 255
    return video_frames


if __name__ == "__main__":

    ######################## INPUTS ######################
    video_path = r'C:\Users\Cristian Stanciu\Downloads\demo\data\videos\id50_id54_stanga_jos.mp4'  #TODO: add path to video

    # EVM Parameters
    # EVM parameters should be veried until you can see the change in colour for the pulse in real life, without too much noise.

    alpha = 5 # Between 2 and 40
    lowcut_freq = 1 # Frequency that needs to be amplified. Min value
    highcut_freq = 2 # Frequency that needs to be amplified. Ma value
    levels = 6 # Levels in the Gaussian pyramid.
    ######################################################

    face_frames, fs = load_video(video_path, save_face=True, only_face_frames=True)
    full_video = video_resize(face_frames, 4)

    evm_video = video2evm(face_frames, alpha=alpha, lowcut_freq=lowcut_freq,
                          highcut_freq=highcut_freq, fs=fs, levels=levels)

    # Showing eulerian video magnification
    for frame in evm_video:

        frame = bgr2rgb(frame)
        frame = cv2.resize(frame, (400, 400))
        cv2.imshow('img', frame/255)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    # Showing original video, resized
    for frame in full_video:

        frame = frame[:, :, [2, 1, 0]]
        frame = cv2.resize(frame, (400, 400))
        cv2.imshow('img', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
