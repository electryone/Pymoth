#!/usr/bin/env python3

import argparse

from PyTrack import PyTrack


def main(args):
    mot_challenge = PyTrack(args.directory)
    mot_challenge.summary()

    # Show sequences for each set of labels for each video sequence from each data set
    # i.e. <test/train> . <video_name> . <det/gt>
    for _, data_sets in mot_challenge.data().items():
        for _, sequences in data_sets.data().items():
            for _, sequence in sequences.data().items():
                sequence.show(draw=True, width=2, show_ids=True, restrict_fps=True)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", type=str, required=True,
                    help="Path to MOTChallenge directory")
args = parser.parse_args()
main(args)
