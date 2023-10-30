import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, find_peaks


#######################################################################################################
"Changeable parameters"
# choosing the face haar cascade used
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# if True, will use default camera. If false, will load a video from video_address
use_live = True
video_address = None

# The min and max values for possible heart rates should be selected and filtered (to avoid noise).
# Freq is in Hz. 1Hz = 60BPM
lowcut_freq = 1.15 # 1Hz = 60 BPM => 1.15 Hz = 69 BPM
highcut_freq = 2.1 # 1Hz = 60BPM => 1.7 Hz = 102 BPM

# Coor plane to be observed. green provides the best results
choose_colour = 1  # BGR image, so 1=Green

# Heart rate is calculated as a moving average. Bigger window for bigger precision, smaller window for instant results
signal_window = 14  # seconds - Heart rate signal moving window

read_rate = 30  # frames - How often will the pulse be read
max_pulse_raise_per_read = 4  # BPM - How fast can the pulse raise per second
show_steady_pulse = True

# For accuracy of face detection
scaleFactor = 1.15
minNeighbors = 5
minsize = (100, 100)
rectangle_update_frames = 1


######################################################################################################


# Bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Eliptic bandpass filter
# def ellip_bandpass(fs, lowcut, highcut, rp, rs, order=1):
#       b, a = ellip(order, rp, rs, [lowcut/(fs/2), highcut/(fs/2)], btype='bandpass')
#       return b, a
#
#
#       def ellip_bandpass_filter(data, fs, lowcut, highcut, rp, rs, order=1):
#       b, a = ellip_bandpass(fs, lowcut, highcut, rp, rs, order)
#       y = lfilter(b, a, data)
#       return y


def get_frequency(array, fs, count_peaks=False):

    #Windowing the signal for better fft accuracy
    #array = array * np.blackman(len(array))

    T = 1 / fs
    # Preinitialising the filter
    array = np.array(list(np.ones(200)*np.mean(array[0:int(len(array)/3)])) + list(array) )
    filtered_array = butter_bandpass_filter(array, lowcut_freq, highcut_freq, fs=fs, order=4)

    #filtered_array = butter_bandpass_filter(filtered_array, lowcut_freq, highcut_freq, fs=fs, order=1)
    #Zero padding enables you to obtain more accurate amplitude estimates of resolvable signal components
    #adding as many zeros as len of the array
    filtered_array = list(filtered_array[200::]) + list(np.zeros(len(filtered_array)-200))
    array_fft = abs(np.fft.fft(filtered_array))
    #max index, but only from the first half
    max_index = np.argmax(abs(array_fft[0:int((len(filtered_array) - 1 +
    len(np.zeros(len(filtered_array)-200))) / 2)]))
    # T = 1/fs
    xf = np.linspace(0.0, 1.0 / (T), len(array_fft))
    # plt.figure()
    # plt.plot(filtered_array)
    # plt.show()
    # plt.title('ord2+1')
    if not count_peaks:
        return xf[max_index]

    if count_peaks:
        peaks_locations = find_peaks(filtered_array)[0]
        before = -10000
        correct_peaks = 0
        for i in peaks_locations:
            if i - before >= 12:
                correct_peaks += 1
                before = i
            else:
                before = before

        return fs / ((peaks_locations[-1]-peaks_locations[0]) / correct_peaks)


def main():



    if use_live:
        cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FPS, 16)
    else:
        cap = cv2.VideoCapture(video_address)

    fs = int(cap.get(5))  # frecventa de esantionare
    print(fs)

    T = 1 / fs
    f_square_width_vector = []
    frame_nr = 0
    first_10 = True
    ret = True
    colour_vector = []
    total_frames = 0
    all_bpm = []
    nr_of_window_samples = int(signal_window * fs)
    first = True
    max_nr_of_pixels = 200000  # Too many pixels can not be processed in real time
    # face_frames = []
    freq = None
    freq_show = None
    last_freq = None
    show_rect = False
    pulse_vector = []

    while ret:

        ret, img_init = cap.read()
        if ret is False:
            break

        if first:
            nr_of_pixels = img_init.shape[0] * img_init.shape[1]
            print(nr_of_pixels, img_init.shape[0], '*' , img_init.shape[1])
            if nr_of_pixels > max_nr_of_pixels:
                resize_factor = max_nr_of_pixels / nr_of_pixels
            else:
                resize_factor = 1

            first = False

        img = cv2.resize(img_init, None, fx=resize_factor, fy=resize_factor)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors, minSize=minsize)

        if len(faces) == 0:
            img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            continue

        if not first_10:
            if len(faces) == 1:
                total_frames += 1

        # Using the first 10 for rectangle.
        if len(faces) > 1:
            faces = [faces[0]]
        if len(faces) == 1:
            frame_nr += 1
            f_square_width = int(faces[0][2] / 4)
            # f_square_width = faces[0][2] / 4
            if frame_nr == 10:
                first_10 = False

            for (x, y, w, h) in faces:

                roi_gray = cv2.resize(gray[y:y + h, x:x + w], (96, 96), interpolation=cv2.INTER_CUBIC)
                roi_color = cv2.resize(img[y:y + h, x:x + w], (96, 96), interpolation=cv2.INTER_CUBIC)
                # face_frames.append(roi_color)

                if total_frames % rectangle_update_frames == 0:
                    saved_values = x, y, w, h
                    show_rect = True

                if show_rect:
                    cv2.rectangle(img, (saved_values[0], saved_values[1]),
                                  (saved_values[0] + saved_values[2], saved_values[1] + saved_values[3]), (255, 255, 0), 2)

            if not first_10:

                forehead = [[saved_values[0] + int(saved_values[2] / 2 - int(f_square_width / 2)),
                             saved_values[1]+ int(saved_values[3] / 12),
                             f_square_width,
                             int(f_square_width / 2)]]

                forehead_square_pixels = 1

                for (a, b, c, d) in forehead:

                    cv2.rectangle(img, (a, b), (a + c, b + d), (255, 0, 0), forehead_square_pixels)

                colour_values = img[forehead[0][1] + forehead_square_pixels:forehead[0][1] +
                                    forehead[0][3] - forehead_square_pixels,
                                    forehead[0][0] + forehead_square_pixels:forehead[0][0] +
                                    forehead[0][2] - forehead_square_pixels,
                                    choose_colour]

                f_colour = np.mean(colour_values)
                colour_vector.append(f_colour)

            if total_frames > nr_of_window_samples + 20 and total_frames % read_rate == 0:

                # heart_beat_frequency = get_frequency(colour_vector[-nr_of_window_samples:])
                total_freqs = 0
                for i in range(1, 21):
                    total_freqs += get_frequency(colour_vector[-(nr_of_window_samples + i):-abs(i)], fs=fs)

                print('BPM:', end='')
                print(total_freqs / 20 * 60, end=' Frame:')
                print(total_frames)
                all_bpm.append(total_freqs / 20 * 60)

                # Printing pulse on image
                font = cv2.FONT_HERSHEY_SIMPLEX
                freq = int((total_freqs / 20 * 60))

                if show_steady_pulse:
                    if last_freq is not None:
                        mpr_pulse = max_pulse_raise_per_read + np.random.randint(4)
                        if freq - last_freq > mpr_pulse:
                            freq_show = last_freq + mpr_pulse
                            last_freq = freq_show
                        elif freq - last_freq < -mpr_pulse:
                            freq_show = last_freq - mpr_pulse
                            last_freq = freq_show
                        else:
                            freq_show = freq
                            last_freq = freq
                    else:
                        last_freq = freq
                        freq_show = freq

                    print('BPM shown:', freq_show)

                else:

                    freq_show = freq

            if freq_show:
                pulse_vector.append(freq_show)
                cv2.putText(img, str(freq_show), (0, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


        #print(img.shape)
        img = cv2.resize(img, (img.shape[1] * 3, img.shape[0] * 3))
        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    print('Finished live heart rate analysis')
    #################################################################################

    # Further analysis for the full signal

    # Adding dummy signal
    colour_vector = np.array(list(np.ones(250) * np.nanmean(colour_vector[0:int(len(colour_vector) / 10)]))
                             + list(colour_vector))
    plt.figure(1)
    plt.title('Evolutia valorii semnalului de culoare verde in timp')
    plt.xlabel('Numarul de cadre')
    plt.ylabel('Amplitudine')
    plt.plot(colour_vector)

    xf = np.linspace(0.0, 1.0 / (T), len(colour_vector) - 250)
    filtered_colour_vector = butter_bandpass_filter(colour_vector, lowcut_freq, highcut_freq, fs=fs,
                                                    order=4)
    # filtered_colour_vector = butter_bandpass_filter(filtered_colour_vector, lowcut_freq, highcut_freq, fs=fs, order=1)

    filtered_colour_vector = filtered_colour_vector[250::]
    colour_vector_fft = abs(np.fft.fft(filtered_colour_vector))
    max_index = np.argmax(abs(colour_vector_fft[0:int((total_frames - 1) / 2)]))

    print('Freq:', end='')
    print(xf[max_index])
    print('Heart BPM: ', end='')
    print(xf[max_index] * 60)


    plt.figure(2)
    plt.xlabel('Frecventa[Hz]')
    plt.ylabel('Amplitudine')
    plt.title('Reprezentarea in frecventa a semnalului de interes ')
    plt.plot(xf, colour_vector_fft)


    plt.figure(3)
    plt.title('Evolutia semnalului de culoare verde FILTRAT in timp')
    plt.xlabel('Numarul de cadre')
    plt.ylabel('Amplitudine')
    plt.plot(filtered_colour_vector)

    #################################################################################
    # For breathing measurement:
    # colour_vector = colour_vector[250::]

    plt.figure(4)
    plt.xlabel('Time')
    plt.ylabel('BPM')
    plt.title('Evolutia pulsului')
    plt.plot(pulse_vector)

    plt.figure(5)
    plt.title('Ritm respirator:Evolutia semnalului de culoare verde in timp')
    plt.xlabel('Numarul de cadre')
    plt.ylabel('Amplitudine')
    plt.plot(colour_vector)

    # adaugarea a unui semnal dummy mai mare
    colour_vector = np.array(list(np.ones(400) * np.nanmean(colour_vector[250:250 + int(len(colour_vector) // 5)]))
                             + list(colour_vector))

    lowcut_freq_breathing = 0.2
    highcut_freq_breathing = 0.4
    xf = np.linspace(0.0, 1.0 / (T), len(colour_vector) - 250)

    filtered_colour_vector = butter_bandpass_filter(colour_vector, lowcut_freq_breathing, highcut_freq_breathing, fs=fs, order=3)
    # filtered_colour_vector = butter_bandpass_filter(filtered_colour_vector, lowcut_freq, highcut_freq, fs=fs, order=1)
    filtered_colour_vector = filtered_colour_vector[250::]
    colour_vector_fft = abs(np.fft.fft(filtered_colour_vector))
    max_index = np.argmax(abs(colour_vector_fft[0:int((total_frames - 1) / 2)]))
    print('Freq:', end='')
    print(xf[max_index])
    print('RPM: ', end='')
    print(xf[max_index] * 60)

    plt.figure(5)
    plt.xlabel('Frecventa[Hz]')
    plt.ylabel('Amplitudine')
    plt.title('Ritm respirator:Reprezentarea in frecventa a semnalului de interes ')
    plt.plot(xf, colour_vector_fft)


    plt.figure(6)
    plt.title('Ritm respirator:Evolutia semnalului de culoare verde FILTRAT in timp')
    plt.xlabel('Numarul de cadre')
    plt.ylabel('Amplitudine')
    plt.plot(filtered_colour_vector)


    plt.show()


if __name__ == "__main__":

    main()
