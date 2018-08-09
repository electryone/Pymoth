#!/usr/bin/env python3

import argparse

from pymoth import Sequence


def main(args):
    sequence = Sequence()
    sequence.load_frames(args.img, args.data, args.info)
    sequence.show(draw=True, show_ids=True, width=2)


parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=False,
                    help="Path to sequence directory")
parser.add_argument("--img", type=str, required=True,
                    help="Path to directory of sequence images")
parser.add_argument("--data", type=str, required=True,
                    help="Path to sequence detection or ground truth data")
parser.add_argument("--info", type=str, required=True,
                    help="Path to sequence info file")
args = parser.parse_args()

if args.dir is not None:
    args.img = "%s/%s" % (args.dir, args.img)
    args.data = "%s/%s" % (args.dir, args.data)
    args.info = "%s/%s" % (args.dir, args.info)

main(args)
