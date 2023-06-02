import wave
import numpy as np
import matplotlib.pyplot as plt
import stumpy
import os
import sys
from moviepy.editor import VideoFileClip 
import tempfile
from pathlib import Path

def gopro_sync(left, right, trimmed_left, trimmed_right):

    left = Path(left)
    right = Path(right)
    trimmed_left = Path(trimmed_left)
    trimmed_right = Path(trimmed_right)

    # Create a temporary directory to store the audio files
    with tempfile.TemporaryDirectory() as tmpdirname:
        
        left_audio = Path(tmpdirname) / Path("left.wav")
        right_audio = Path(tmpdirname) / Path("right.wav")

        left = VideoFileClip(str(left))
        left.audio.write_audiofile(left_audio)
        left.close()

        right= VideoFileClip(str(right))
        right.audio.write_audiofile(right_audio)
        right.close()

        # Begin uploading audio from file into a numpy array

        # left_wav = wave.open(str(left_audio), "rb")
        # right_wav = wave.open(right_audio, "rb")

        # print(np.shape(left_wav))

def argument_parser(args):

    # make sure flags are used correctly
    acceptable_args = ["--left", "--out", "--right"]
    args_1 = args[1:3]
    args_2 = args[3:5]
    args_3 = args[5:7]

    if args_1[0] not in acceptable_args:
        print(f"{args_1[0]} is not an acceptable flag. Please use the flags --left, --right, --out")
        exit()
    elif args_2[0] not in acceptable_args:
        print(f"{args_2[0]} is not an acceptable flag. Please use the flags --left, --right, --out")
        exit()
    elif args_3[0] not in acceptable_args:
        print(f"{args_3[0]} is not an acceptable flag. Please use the flags --left, --right, --out")
        exit()

    if len(set([args_1[0], args_2[0], args_3[0]])) != 3:
        print("Duplicate flags. Please try again")
        exit()

    # Getting all the paths organized into the appropriate variables
    path_left, path_right, path_out = ("", "", "")

    if (args_1[0] == "--left"):
        path_left = args_1[1]
        if (args_2[0] == "--right"):
            path_right = args_2[1]
            path_out = args_3[1]
        else:
            path_out = args_2[1]
            path_right = args_3[1]

    elif (args_1[0] == "--right"):
        path_right = args_1[1]
        if (args_2[0] == "--left"):
            path_left = args_2[1]
            path_out = args_3[1]
        else:
            path_out = args_2[1]
            path_left = args_3[1]
    
    elif (args_1[0] == "--out"):
        path_out = args_1[1]
        if (args_2[0] == "--left"):
            path_left = args_2[1]
            path_right = args_3[1]
        else:
            path_right = args_2[1]
            path_left = args_3[1]        

    # Ensure paths exist
    if not os.path.isfile(path_left):
        print(f"{path_left} is not an acceptable path for flag --left, which should point to a file.")
        exit()
    elif not os.path.isfile(path_right):
        print(f"{path_right} is not an acceptable path for flag --right, which should point to a file.")
        exit()
    elif not os.path.isdir(path_out):
        print(f"{path_out} is not an acceptable path for flag --out, which should point to a directory.")
        exit()

    # Ensure that left and right files are mp4s
    if (path_left[-4:] != ".mp4" and path_left[-4:] != ".MP4"):
        print("File for left is not an mp4. Please try again.")
        exit()
    elif (path_right[-4:] != ".mp4" and path_right[-4:] != ".MP4"):
        print("File for right is not an mp4. Please try again.")
        exit()

    return path_left, path_right, path_out

# Check number of arguments is correct
total_arguments = len(sys.argv)
if (total_arguments != 7):

    print("Incorrect number of arguments. Please ensure there are 6 arguments after the file name. For example: ")
    print("python3 goprosync.py --left left.mp4 --right right.mp4 --out ./final/")
    exit()

# Ensure that all arguments are made correctly
left_path, right_path, out_path = argument_parser(sys.argv)
left_trimmed = os.path.join(out_path, "left_trimmed.mp4")
right_trimmed = os.path.join(out_path, "right_trimmed.mp4")

gopro_sync(left=left_path, right=right_path, trimmed_left=left_trimmed, trimmed_right=right_trimmed)




