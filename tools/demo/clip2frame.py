import cv2
import numpy as np
import argparse

def clip2frame(clip, keyframe):

    '''
    Author: Francesco Maria Turno

    Method that extracts desired keyframe from movie clip (e.g.: LSMDC dataset)

    INPUT
    -----

    * clip : file path where movie clip is located (string)
    * keyframe : frame of interest (positive int)

    OUTPUT
    ------

    * frame: extracted image (Most Descriptive Frame) in .jpg format (array)
    '''

    cam = cv2.VideoCapture(clip + '.mp4')
    cam.set(cv2.CAP_PROP_POS_FRAMES, keyframe-1) # this step is crucial
    ret, frame = cam.read()

    clip_name = clip.split(".")[0]
    # print(clip)

    output_file = clip_name + '_frame_' + str(keyframe) + '.jpg'
    print(output_file)

    print("Writing..." +output_file)

    cv2.imwrite(output_file, frame)

    print('Frame_' + str(keyframe) + '.jpg (MDF) has been exported successfully!')

    cam.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Take arguments as input")
    parser.add_argument("--path", metavar="PATH",
                        help="file path")
    parser.add_argument("--keyframe", metavar="KEYFRAME", help="frame of interest")
    args = parser.parse_args()
    print("Loading... " + args.path)
    clip2frame(args.path, int(args.keyframe))

if __name__ == "__main__":
    main()
