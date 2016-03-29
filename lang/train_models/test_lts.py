#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is able to test the performance of a LTS rules model, given
a lexicon.

Author: Sergio Oller, 2016
"""
from __future__ import unicode_literals
from __future__ import print_function

from common import read_align, read_lts, process_lts, test_lts

import argparse


def load_and_test_lts(align_fn, lts_rules_fn):
    align = read_align(align_fn)
    lts_raw = read_lts(lts_rules_fn)
    lts = process_lts(lts_raw)
    accuracy = test_lts(align, lts)
    print("LTS word accuracy on train set (cmulex expected ~60%): {:.1%}".
          format(accuracy))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Tests a LTS rules model using an align file.')
    parser.add_argument('--align_file', required=True,
                        help='Align file to test the LTS model against.')
    parser.add_argument('--lts_rules', required=True,
                        help='LTS rules scm file.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    align_fn = args.align_file
    lts_rules_fn = args.lts_rules
    load_and_test_lts(align_fn, lts_rules_fn)
