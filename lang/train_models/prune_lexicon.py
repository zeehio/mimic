#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is able to prune a lexicon with entries that are predicted
correctly by an LTS model.

Author: Sergio Oller, 2016
"""
from __future__ import unicode_literals
from __future__ import print_function

import argparse

from common import (read_lexicon, read_lts, process_lts, prune_lexicon,
                    write_lex)


def load_and_prune_lex(lexicon_fn, lts_rules_fn, output_pruned_lex_fn):
    lexicon = read_lexicon(lexicon_fn)
    lts_raw = read_lts(lts_rules_fn)
    lts = process_lts(lts_raw)
    pruned_lex = prune_lexicon(lexicon, lts)
    write_lex(pruned_lex, output_pruned_lex_fn, flattened=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Helpers to create lexicon and LTS rules.')
    parser.add_argument('--lexicon', required=True,
                        help='Path to the lexicon file in Festival format')
    parser.add_argument('--lts_rules', required=True,
                        help='LTS rules scm file.')
    parser.add_argument('--output', required=True,
                        help='Lexicon output file, only with words not' +
                             'predicted correctly by the LTS rules')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    lexicon_fn = args.lexicon
    lts_rules_fn = args.lts_rules
    output_pruned_lex_fn = args.output
    load_and_prune_lex(lexicon_fn, lts_rules_fn, output_pruned_lex_fn)
