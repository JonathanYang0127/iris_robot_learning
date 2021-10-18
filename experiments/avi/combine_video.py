import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video-path", type=str)
args = parser.parse_args()

def parse_vid(s):
    period_idx = s.index(".")
    return int(s[:period_idx])

with open("inputs.txt", "w+") as file:
    for dir in sorted(os.listdir(args.video_path), key=lambda x: int(x)):
        for vid in sorted(os.listdir("{}/{}".format(args.video_path, dir)), key=parse_vid):
            file.write("file '{}'\n".format(os.path.join(args.video_path, dir, vid)))
os.system("ffmpeg -f concat -safe 0 -i inputs.txt -c copy {}".format(os.path.join(args.video_path, "output.mp4")))
os.system("rm inputs.txt")
