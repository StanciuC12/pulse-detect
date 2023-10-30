
These scripts are can be used to read the pulse (BPM) of a person with a non-contact video method.

1. pulse-from-video.py

This script can run in real time (with the default webcam) or offline (uploaded video).

Parameters: In the first lines of the code, the program parameters can be changed:

- face_cascade: what face cascade classifier to use. Here, we use haar cascades because they are very efficient from a computational standpoint
a haar cascade xml file is included

- use_live: if True, the script will run live, using the camera.
if False, the script will run on a video, found at video_address

- The human pulse is usually between some values. The script filters the rest of the frequencies, because they are noise.
lowcut_freq = lowcut frequency of the bandpass filter, in Hz. 1 Hz = 60 BPM.
recommended values: lowcut freq: 1-1.2 Hz;  highcut_freq: 1.6-2.5 Hz

- choose_color: what color plane to use for pulse detection. Default = 1 = Green (BGR)

-signal_window: the pulse is obtained using a moving average. That moveing average is donem inside a window.
signal_window = 5 means that we use 5 seconds of video at a time to determine pulse. Smaller value = more instant pulse reading.
bigger value = more accurate pulse reading

- read_rate: how often we calculate the pulse BPM value. default = 1 (once every 1 frame)

-scaleFactor, minNeighbours = haar cascade classifier parameters


Running:

When the program is started, it will show the video feed and it will put the face in a bounding box.
 The program can only work with 1 face at a time! After signal_window seconds, the first pulse reading will
 be done. The BPM value should appear in the top left corner, with red.

 To stop the program, press Escape. After stopping the program, 6 plots will appear: 3 for pulse and 3 fro breathing.
 1st: raw signal; 2nd: fft of wignal; 3rd: filtered signal


 Tips and tricks:
 - The program works best at 60+ FPS. 30 is file, less is very noisy.
 - The heart rate and breathing rate is calculated by looking at the skin colour (using photoplestimography).
 Variations in the skin colour can not be seen by the naked eye, but they can be seen by a camera.
 - HD cameras work better, but they do not make a huge difference.
 - The program caps the maximum number of pixels in the video, so it can work in real time. If it is laggy or
 the results do not make sense, the camera may work under the default frame rate.


 2. eulerian_video_magnification.py


 Eulerian video magnification amplifies changes in images.The amplified changes help in seeting thing in video with ease.
 EG: we can magnify frequencies between Hz and 2Hz to see colour variations in the skin with the naked eye.
 Another benefit is that the signal has a bigger amplitude and less noise.
  The script does not work in 30 FPS as it is, but it works fine offline.

  In the __main__ part of the script, there are a few input parameters:

  - video path

  - EVM parameters:

    alpha = 5  - Between 2 and 40 - how much to amplify the selected frequencies in the video
    lowcut_freq = 1 - Frequency that needs to be amplified. Min value
    highcut_freq = 2 - Frequency that needs to be amplified. Ma value
    levels = 6 - Levels in the Gaussian pyramid. Should apply more if the image is more HD.

   - The parameters should be varied offline until you can see the magnified frequencies with the naked eye,
   without much noise.

   - A video with frequencies amplified by EVM should work better when detecting pulse with the other script.


   3. video_functions2 : help functions for the other scripts

   4. requirements.txt: the program runs with the library versions in the file. If should also work with newer ones, but
   it is not tested.