# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 01:18:41 2016

@author: sergio
"""

import os
from subprocess import call
from functools import partial
import numpy as np
from .utils import progress_bar
from .scheme import parse
from .common import eval_tree


def print_lts_desc(featsnp, feat_names, lts_desc_fn):
    with open(lts_desc_fn, "w") as fd:
        print("(", file=fd)
        for i, feat_name in enumerate(feat_names):
            feat_values = sorted(np.unique(featsnp[:, i]))
            print("(" + feat_name, file=fd)
            print(" ".join(feat_values), file=fd)
            print(")", file=fd)
        print(")", file=fd)


def call_wagon(input_fn, output_fn, stop, lts_desc_fn, wagon_path):
    wagon_args = ["-data", input_fn, "-test", input_fn, '-desc',
                  lts_desc_fn, "-stop", str(stop), "-output", output_fn]
    call([wagon_path] + wagon_args)
    return


def build_letter(letter, featsnp, feat_central, stop,
                 scratchdir, lts_desc_fn, wagon_path):
    feats_np_letter = featsnp[featsnp[:, feat_central] == letter, :]
    input_fn = os.path.join(scratchdir, "ltsdataTRAIN." + letter + ".feats")
    output_fn = os.path.join(scratchdir, "lts." + letter + ".tree")
    np.savetxt(fname=input_fn, X=feats_np_letter, fmt="%s")
    call_wagon(input_fn, output_fn, stop, lts_desc_fn, wagon_path)


def build_lts(letters, featsnp, feat_names, feat_central, stop,
              scratchdir, wagon_path):
    lts_desc_fn = os.path.join(scratchdir, "ltsLTS.desc")
    print_lts_desc(featsnp, feat_names, lts_desc_fn)
    build_let = partial(build_letter, featsnp=featsnp,
                        feat_central=feat_central, stop=3,
                        scratchdir=scratchdir, lts_desc_fn=lts_desc_fn,
                        wagon_path=wagon_path)
    for i, letter in enumerate(letters):
        progress_bar(i, len(letters))
        build_let(letter)


def read_tree(filename):
    """ This parses a lts_scratch/*.tree file. It contains the decision tree
    trained for a letter."""
    with open(filename, "rt") as fd:
        output = []
        for line in fd.readlines():
            # comment (license, author...)
            if line.startswith(";"):
                continue
            output.append(line)
        # A first parenthesis is added because the (set! line has a
        # parenthesis that we want to keep.
        all_rules = " ".join(output)
        lts = parse(all_rules)
    return lts

def merge_models(letters, scratchdir, output_fn):
    output = list()
    for letter in letters:
        tree = read_tree(os.path.join(scratchdir, "lts." + letter + ".tree"))
        output.append((letter, tree))
    return output

def test_lts():
    pass

# (define (merge_models name filename allowables)
# "(merge_models name filename)
# Merge the models into a single list of cart trees as a variable
# named by name, in filename."
#  (require 'cart_aux)
#  (let (trees fd)
#    (set! trees nil)
#    (set! lets (mapcar car allowables))
#    (while lets
#      (if (probe_file (format nil "lts.%s.tree" (car lets)))
#          (begin
#            (format t "%s\n" (car lets))
#            (set! tree (car (load (format nil "lts.%s.tree" (car lets)) t)))
#            (set! tree (cart_simplify_tree2 tree nil))
#            (set! trees
#                  (cons (list (car lets) tree) trees))))
#      (set! lets (cdr lets)))
#    (set! trees (reverse trees))
#    (set! fd (fopen filename "w"))
#    (format fd ";; LTS rules \n")
#    (format fd "(set! %s '(\n" name)
#    (mapcar
#     (lambda (tree) (pprintf tree fd))
#     trees)
#    (format fd "))\n")
#    (fclose fd))
# )
