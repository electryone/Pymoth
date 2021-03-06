#!/usr/bin/env python3

import argparse

from pymoth import DataSet


def main(args):
    mot_challenge = DataSet(args.directory)
    mot_challenge.summary()

    # Show sequences for each set of labels for each video sequence from each data set
    # i.e. <test/train> . <video_name> . <det/gt>
    for _, data_sets in mot_challenge.get().items():
        for _, sequences in data_sets.get().items():
            for _, sequence in sequences.get().items():
                sequence.show(draw=True, width=2, show_ids=True, restrict_fps=True)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--directory", type=str, required=True,
                    help="Path to MOTChallenge directory")
args = parser.parse_args()
main(args)
