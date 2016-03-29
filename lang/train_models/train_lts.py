#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script is able to filter a lexicon removing short words not suitable for
LTS rules training.

Author: Sergio Oller, 2016
"""
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import copy
from collections import defaultdict

from common import read_lexicon, write_lex


def parse_args():
    parser = argparse.ArgumentParser(description='Helper to train LTS rules.')
    parser.add_argument('--lexicon', required=True,
                        help='Path to the lexicon file in Festival format')
    parser.add_argument('--output', required=True,
                        help='Lexicon output file, with some words filtered' +
                             'based on LTS training requirements')
    parser.add_argument('--minlength', type=int, default=4,
                        help='Exclude lexicon entries shorter than minlength')
    parser.add_argument('--lower', action='store_const',
                        const=True, default=True,
                        help='Lower case lexicon entries')
    args = parser.parse_args()
    return args


def filter_lexicon(lexicon, minlength=4, lower=True):
    filtered_lex = defaultdict(list)
    for word in lexicon.keys():
        word_filtered = word
        if minlength is not None and len(word) < minlength:
            continue
        if lower is True:
            word_filtered = word_filtered.lower()
        filtered_lex[word_filtered] = lexicon[word]
    return filtered_lex


def load_and_filter_lex_for_lts(lexicon_fn, filtered_lex_fn=None,
                                minlength=4, lower=True):
    """ We load the raw lexicon and return a lexicon with words with more than
        three characters. Ideally we could leave nouns, verbs and adjectives as
        the unknown words we will find will probably belong to one of those
        categories.
        See: http://www.festvox.org/docs/manual-2.4.0/festival_13.html#Building-letter-to-sound-rules
    """
    lexicon = read_lexicon(lexicon_fn)
    filtered_lex = filter_lexicon(lexicon, minlength=minlength, lower=lower)
    if filtered_lex_fn is not None:
        write_lex(filtered_lex, filtered_lex_fn, flattened=True)
    return filtered_lex


def valid_pair(phone, letter, pl_table):
    if letter in pl_table.keys() and phone in pl_table[letter].keys():
        return True
    return False


def valid_pair_e(phone, nphone, letter, pl_table):
    if (letter in pl_table.keys() and
            (phone + "-" + nphone) in pl_table[letter].keys()):
        return True
    return False


def find_all_aligns(phones, letters, pl_table):
    """Find all feasible alignments."""
    r = []
    if (len(phones) == 1 and len(letters) == 1 and
            phones[0] == letters[0] and phones[0] == '#'):
        return [[('#', '#')]]
    if valid_pair('_epsilon_', letters[0], pl_table):
        all_left = find_all_aligns(phones, letters[1:], pl_table)
        r += [[('_epsilon_', letters[0])] + x for x in all_left]
    if valid_pair(phones[0], letters[0], pl_table):
        all_left = find_all_aligns(phones[1:], letters[1:], pl_table)
        r += [[(phones[0], letters[0])] + x for x in all_left]
    # Hmm, change this to always check doubles
    try:
        if (len(phones) > 1 and
                valid_pair_e(phones[0], phones[1], letters[0], pl_table)):
            all_left = find_all_aligns(phones[2:], letters[1:], pl_table)
            two_phones = phones[0] + "-" + phones[1]
            r += [[(two_phones, letters[0])] + x for x in all_left]
    except:
        print(phones)
        print(letters)
        raise
    return r


def cummulate(phone, letter, pl_table):
    "record the alignment of this phone and letter."
    if (phone == letter or (phone != "#" and letter != "#")):
        score = 1.0
        if phone == "_epsilon_":
            score = 0.1
        pl_table[letter][phone] += score
    return


def cummulate_aligns(all_aligns, pl_table):
    for align in all_aligns:
        for (phone, letter) in align:
            cummulate(phone, letter, pl_table)
    return


def cummulate_pairs(lexicon, allowables):
    failed_aligns = 0
    count_all_aligns = 0
    failed_list = []
    # initialize pl_table of letter-phone counts
    pl_table = defaultdict(lambda: defaultdict(int))
    for letter, phones in allowables.items():
        for phone in phones:
            pl_table[letter][phone] = 0
    percent_done = 0  # percent_* is used for printing progress information
    percent_step = 0.05
    for i, (word, heteronyms) in enumerate(sorted(lexicon.items())):
        if i/len(lexicon.keys()) > percent_done:
            print("[{0:.0%}] ENTRY: {1} {2}".format(i/len(lexicon.keys()),
                                                    i, word))
            percent_done += percent_step
        for heteronym in heteronyms:
            # pos = heteronym[0]
            # phones_in_syl = heteronym[1]
            phones = heteronym[2]
            # add word boundaries:  # enworden(wordexplode(word))
            bound_phones = ['#'] + phones + ['#']
            bound_word = ['#'] + list(word) + ['#']
            try:
                all_aligns = find_all_aligns(bound_phones,
                                             bound_word,
                                             pl_table)
            except:
                print(bound_word)
                print(bound_phones)
                raise
            if len(all_aligns) == 0:
                failed_aligns += 1
                failed_list.append((word, " ".join(phones)))
            cummulate_aligns(all_aligns, pl_table)
            count_all_aligns += 1
    print("\n".join([": ".join(x) for x in failed_list]))
    print("Failed aligns: {}/{}".format(failed_aligns, count_all_aligns))
    return pl_table


def normalise_table(pl_table):
    "Change scores into probabilities."
    pl_table_norm = copy.deepcopy(pl_table)
    for letter in pl_table_norm.keys():
        all_counts = sum(pl_table_norm[letter].values())
        for phone in pl_table_norm[letter].keys():
            if all_counts == 0:
                pl_table_norm[letter][phone] = 0
            else:
                pl_table_norm[letter][phone] /= all_counts
    return pl_table_norm


def save_pl_table(pl_table, output_fn):
    with open(output_fn, "wt") as fd:
        print("(set! pl-table'", file=fd)
        for ilet, letter in enumerate(sorted(pl_table.keys())):
            if ilet == 0:
                print("(", file=fd, end="")
            else:
                print(" ", file=fd, end="")
            print("({}".format(letter), file=fd)
            num_phones = len(pl_table[letter].keys())
            for iphone, (phone, value) in \
                    enumerate(sorted(pl_table[letter].items())):
                myline = "  (" + phone + " . " + str(value) + ")"
                if iphone == num_phones-1:
                    myline += ")"
                print(myline, file=fd)
        print("))", file=fd)


def load_and_cummulate_pairs(lexicon_fn, allowables, output_fn):
    lexicon = read_lexicon(lexicon_fn, is_flat=True)
    print("Find probabilities of letter-phone pairs")
    pl_table = cummulate_pairs(lexicon, allowables)
    pl_table_norm = normalise_table(pl_table)
    save_pl_table(pl_table_norm, output_fn)
    return pl_table_norm

ALLOWABLES = {'a': ['_epsilon_', 'aa', 'aa1', 'aa0', 'ax', 'ax1', 'ax0', 'eh',
                    'eh1', 'eh0', 'ah', 'ah1', 'ah0', 'ae', 'ae1', 'ae0', 'ey',
                    'ey1', 'ey0', 'ay', 'ay1', 'ay0', 'er', 'er1', 'er0',
                    'y-ax0', 'y-ah1', 'y-ah0', 'aw', 'aw1', 'aw0', 'ao',
                    'ao1', 'ao0', 'ih', 'ih1', 'ih0', 'w-ax0', 'w-ah1',
                    'w-ah0', 'ow', 'ow1', 'ow0', 'w-ey', 'w-ey1', 'ey0',
                    'iy', 'iy1', 'iy0'], 'h': ['_epsilon_', 'hh'],
              'b': ['_epsilon_', 'b', 'p'],
              'c': ['_epsilon_', 'k', 'ch', 's', 'sh', 't-s'],
              'd': ['_epsilon_', 'd', 't', 'jh'],
              'e': ['_epsilon_', 'ih', 'ih1', 'ih0', 'ax', 'ax1', 'ax0', 'iy',
                    'iy1', 'iy0', 'er', 'er1', 'er0', 'ax', 'ah1', 'ah0', 'eh',
                    'eh1', 'eh0', 'ey', 'ey1', 'ey0', 'uw', 'uw1', 'uw0', 'ay',
                    'ay1', 'ay0', 'ow', 'ow1', 'ow0', 'y-uw', 'y-uw1', 'y-uw0',
                    'oy', 'oy1', 'oy0', 'aa', 'aa1', 'aa0'],
              'f': ['_epsilon_', 'f'],
              'g': ['_epsilon_', 'g', 'jh', 'zh', 'k', 'f'],
              'i': ['_epsilon_', 'iy', 'iy1', 'iy0', 'ax', 'ax1', 'ax0', 'ih',
                    'ih1', 'ih0', 'ah', 'ah1', 'ah0', 'ax', 'ah1', 'ah0',
                    'ay', 'ay1', 'ay0', 'y', 'aa', 'aa1', 'aa0', 'ae',
                    'ae1', 'ae0', 'w-ax0', 'w-ah1', 'w-ah0', 'eh', 'eh1',
                    'eh0', 'er', 'er0', 'er1'],
              'j': ['_epsilon_', 'jh', 'y', 'hh', 'zh'],
              'k': ['_epsilon_', 'k'],
              'l': ['_epsilon_', 'l', 'ax-l', 'y', 'ax0-l'],
              'm': ['_epsilon_', 'm', 'ax-m', 'm-ax0', 'ax0-m', 'm-ax0',
                    'm-ae', 'm-ae1', 'm-ae0', 'm-ih', 'm-ih0'],
              'n': ['_epsilon_', 'n', 'ng', 'n-y'],
              'o': ['_epsilon_', 'ax', 'ax0', 'ah1', 'ah0', 'ao', 'ao1', 'ao0',
                    'ow', 'ow1', 'ow0', 'uw', 'uw1', 'uw0', 'er', 'er1', 'er0',
                    'aa', 'aa1', 'aa0', 'aw', 'aw1', 'aw0', 'oy', 'oy1', 'oy0',
                    'uh', 'uh1', 'uh0', 'w', 'w-ax0', 'w-ah1', 'w-ah0', 'aa',
                    'aa1', 'aa0', 'ih', 'ih1', 'ih0', 'ae', 'ae1', 'ae0'],
              'p': ['_epsilon_', 'p', 'f'],
              'q': ['_epsilon_', 'k'],
              'r': ['_epsilon_', 'r', 'er1', 'er', 'er0'],
              's': ['_epsilon_', 's', 'sh', 'z', 'zh', 'ch'],
              't': ['_epsilon_', 't', 'th', 'sh', 'ch', 'dh', 'd', 's', 'zh'],
              'u': ['_epsilon_', 'ax', 'ax0', 'ah', 'ah1', 'ah0', 'uw', 'uw1',
                    'uw0', 'er', 'er1', 'er0', 'uh', 'uh1', 'uh0', 'y-uw',
                    'y-uw1', 'y-uw0', 'ax-w', 'ah1-w', 'ah0-w', 'y-er',
                    'y-er1', 'y-er0', 'y-ax', 'y-ax0', 'y-ah1', 'y-ah0',
                    'w', 'ih', 'ih1', 'ih0', 'ao', 'ao1', 'ao0', 'eh',
                    'eh1', 'eh0', 'y-uh', 'y-uh1', 'y-uh0'],
              'v': ['_epsilon_', 'v', 'f'],
              'w': ['_epsilon_', 'w', 'v', 'f'],
              'x': ['_epsilon_', 'k-s', 'g-z', 'ng-z', 'k-sh', 'z',
                    'g-zh', 'zh'],
              'y': ['_epsilon_', 'iy', 'iy1', 'iy0', 'ih', 'ih1', 'ih0', 'ay',
                    'ay1', 'ay0', 'y', 'ax', 'ax0', 'ah1', 'ah0'],
              'z': ['_epsilon_', 'z', 't-s', 'zh', 's'],
              '#': ['#']}


if __name__ == "__main__":
    args = parse_args()
    lexicon_fn = args.lexicon
    filtered_lex_fn = args.output
    minlength = args.minlength
    lower = args.lower
    print("Filter lexicon removing short words and converting to lower case")
    filtered_lex = load_and_filter_lex_for_lts(lexicon_fn, filtered_lex_fn,
                                               minlength, lower)
    print("Find probabilities of letter-phone pairs")
    output_fn = "lts_scratch/lex-pl-tablesp.scm"
    pl_table = cummulate_pairs(filtered_lex, ALLOWABLES)
    pl_table_norm = normalise_table(pl_table)
    save_pl_table(pl_table_norm, output_fn)
