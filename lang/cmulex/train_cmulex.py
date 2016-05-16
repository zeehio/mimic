#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from train_models.train_lts import (read_lexicon, filter_lexicon, write_lex,
                                    cummulate_pairs,
                                    normalise_table, save_pl_table, align_data,
                                    save_lex_align, build_feat_file)

from train_models.build_lts import build_lts, merge_models


MIMICDIR = os.path.join(os.path.dirname(__file__), "..", "..")


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

WORK_DIR = "festival/lib/dicts/cmu/"
LTS_SCRATCH = os.path.join(WORK_DIR, "lts_scratch")

# The input lexicon:
lexicon_fn = os.path.join(WORK_DIR, "cmudict-0.4.out")

# Filtered lexicon:
# lex_entries is essentially the lexicon with short words removed and
# with lower case. lex_entries is more suitable than the raw lexicon for
# training LTS rules
lex_entries_fn = os.path.join(LTS_SCRATCH, "lex_entries.out")
minlength = 4  # Remove short words (len<4)
lower = True  # Lowercase words

print("Load lexicon in festival format")
lexicon = read_lexicon(lexicon_fn, is_flat=False)
print("Filter lexicon: Removing short words and converting to lower case")
filtered_lex = filter_lexicon(lexicon, minlength=minlength, lower=lower)
write_lex(filtered_lex, lex_entries_fn, flattened=True)

print("Count probabilities of letter-phone pairs")
pl_table = cummulate_pairs(filtered_lex, ALLOWABLES)
pl_table_norm = normalise_table(pl_table)
lex_pl_tablesp_fn = os.path.join(LTS_SCRATCH, "lex-pl-tablesp.scm")
save_pl_table(pl_table_norm, lex_pl_tablesp_fn)  # sort dict by value sorted(d, key=d.get)
print("Align letters with phones")
(good_align, align_failed) = align_data(filtered_lex, pl_table_norm)
lex_align_fn = os.path.join(LTS_SCRATCH, "lex.align")
save_lex_align(good_align, lex_align_fn)
lex_feats_fn = os.path.join(LTS_SCRATCH, "lex.feats")
print("Build feat file")
feats = build_feat_file(good_align, lex_feats_fn)
print("Build LTS models")

import numpy as np
featsnp = np.array(feats)
feat_names = ['Relation.LTS.down.name',
              'p.p.p.p.name ignore',
              'p.p.p.name',
              'p.p.name',
              'p.name',
              'name',
              'n.name',
              'n.n.name',
              'n.n.n.name',
              'n.n.n.n.name ignore',
              'pos ignore']
feat_central = 6  # name

WAGON_PATH = os.path.join(MIMICDIR, "_festsuite",
                          "speech_tools", "bin", "wagon")
all_letters = sorted(set(ALLOWABLES.keys()) - set("#"))
build_lts(all_letters, featsnp, feat_names, feat_central=feat_central,
          stop=3, scratchdir=LTS_SCRATCH, wagon_path=WAGON_PATH)

# Merge not implemented
#lts_model_raw = merge_models(all_letters, LTS_SCRATCH, "")
#from train_models.common import process_lts
#lts_model = process_lts(lts_model_raw)

#from train_models.common import read_align, test_lts
# FIXME: Merge models should save the final model
#al = read_align(lex_align_fn)
#print(test_lts(al, lts_model))

