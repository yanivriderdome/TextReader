import cv2
import pytesseract
import re
import pandas as pd
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
feature_params = dict(maxCorners=50, qualityLevel=0.2, minDistance=0, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def remove_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)


def textExtractor(gray_image):
    thresh = thresholding(gray_image)
    text = pytesseract.image_to_string(thresholding(gray_image), lang='eng', config='--psm 3')
    pattern = r"([0-9]+\.[0-9])"
    find_speed = re.search(pattern, text)

    print(find_speed.group(0))
    if find_speed:
        return find_speed.group(0)

    return -1


def imageTracker(gray_image, previous_gray_image):
    features_to_Track = cv2.goodFeaturesToTrack(previous_gray_image, mask=None, **feature_params)
    next_pos, status, error = cv2.calcOpticalFlowPyrLK(previous_gray_image, gray_image, features_to_Track, None,
                                                       **lk_params)
    good_features_to_Track = features_to_Track[status == 1].astype(float)
    tracked_features = next_pos[status == 1].astype(float)
    diff = good_features_to_Track - tracked_features
    feature_array = [[x[0], x[1], diff[i][0], diff[i][1]] for i, x in enumerate(tracked_features)]
    return feature_array

def get_speed_from_text(text):
    parsed_text = text.split(" ")
    if len(parsed_text) > 2:
        found = False
        for i, x in enumerate(parsed_text):
            if x.find('km') != -1 and i > 0:
                speed = parsed_text[i - 1]
                speed = "".join([char for char in speed if char.isdigit() or char == "."])
                found = True
                break
        if not found:
            try:
                speed = float(speed)
            except:
                return -1
        return speed
    else:
        found = False
        for x in parsed_text:
            if x.find('km') != -1:
                speed = x.split('km')[0]
                found = True
                break
        if not found:
            speed = parsed_text[0]

    return speed

# path_to_video = r"New folder/Blindspot/Harley_Davidson_Left_Blind_Spot_26.mp4"
path_to_video = r"D:\Camera Roll\Alerts\Front Alerts"
output_filename = 'Speeds.csv'
if __name__ == "__main__":

    label = []
    data = []
    columns = ["Black Box Filename", "Black Box Frame Number", "Speed", "text"]
    df = pd.DataFrame(columns=columns)

    if not os.path.exists(output_filename):
        df.to_csv(output_filename, index=False)

    for Filename in os.listdir(path_to_video):
        if os.path.isdir(os.path.join(path_to_video, Filename)):
            continue
        print(Filename)
        video_to_analyse = cv2.VideoCapture(os.path.join(path_to_video, Filename))
        Frame_number = 1
        prevspeed = -1

        while video_to_analyse.isOpened():
            ret, frame = video_to_analyse.read()
            text = ""
            if ret:
                noiseless = remove_noise(frame)
                gray = get_grayscale(noiseless)
                text = pytesseract.image_to_string(thresholding(gray), lang='eng', config='--psm 3')
                speed = get_speed_from_text(text)
                if speed == '' or speed == -1:
                    speed = prevspeed
                else:
                    try:
                        speed = float(speed)
                        if speed < 500:
                            prevspeed = speed
                    except:
                        speed = prevspeed

                ret, frame = video_to_analyse.read()
                ret, frame = video_to_analyse.read()
                ret, frame = video_to_analyse.read()

            else:
                print(speed)

            if not ret:
                break

            data.append([Filename, Frame_number, speed, text])
            Frame_number += 1
            data.append([Filename, Frame_number, speed, text])
            Frame_number += 1
            data.append([Filename, Frame_number, speed, text])
            Frame_number += 1
            data.append([Filename, Frame_number, speed, text])
            Frame_number += 1

            print(Frame_number, speed)
        if len(data) > 0:
            df = pd.DataFrame(data)
            df.columns = columns
            data = []

            df.to_csv(output_filename, mode='a', index=False, header=False)
        # frameToDataframe = {
        #     "Frame Number": "{0}".format(frame_counter),
        #     "Speed": speed
        # }
        # tracked_features = imageTracker(gray, previous_gray)
        # # label.append(speed)
        # # data.append(data)
        #
        # frame_counter += 1
        # prev_gray = gray.copy()
        # print(frameToDataframe)
