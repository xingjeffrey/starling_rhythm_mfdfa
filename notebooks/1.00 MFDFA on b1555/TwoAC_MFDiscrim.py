# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:17:02 2023

@author: jeffrey
"""
import os
import sys
import random
import csv
import shutil
import string
import numpy as np
from pyoperant import utils, components, local, hwio
from pyoperant.behavior import two_alt_choice
import copy
import string


class TwoAC_MFDiscrim(two_alt_choice.TwoAltChoiceExp):
    """
    discriminate between high and low multifractal inter-motif gap shuffle pairs
    """

    def __init__(self, *args, **kwargs):

        super(TwoAC_MFDiscrim, self).__init__(*args, **kwargs)
        # save beginning parameters to reset each session
        self.starting_params = copy.deepcopy(self.parameters)
        self.get_conditions()  # get conditions for trial in block (e.g. ABA, BAB, BBA, ...)
        self.get_motifs()  # builds specific motif sequences (e.g. [A12, B1, A40])
        self.build_block()  # build wav files from the generated sequences


    def get_conditions(self):
        """
        generates a random 100 trial block (l, r, l, r etc)
        """
        self.trial_types = np.matrix('0; 1') ## there are two types of trials
        #
        self.trials = []
        for i in range(100):
            self.trials.append(random.randrange(0, 2, 1)) ## len 100 array of 1 or 0
        self.trial_output = [self.parameters["category_conditions"][i]["class"]
                             for i in self.trials]
                             ## for every trial, have a output array that gives L/R identity

    def get_motifs(self):
        """
        2. generate specific stim sequence e.g. [A12, B1, A40]
        """
        self.motifs = []
        molen = self.parameters["current_available_motifs"]
        left_stims = self.parameters["left_stims"]
        right_stims = self.parameters["right_stims"]
        motif_seq = self.trials

        for i in motif_seq: ## for every trial

            if i == 1: #go right if 1 go left if 0
                thisstim = right_stims[str(random.randrange(0,molen,1))]

            else:
                thisstim = left_stims[str(random.randrange(0,molen,1))]

            self.motifs.append(thisstim)

    def build_block(self):
        """
        Adds stims and trial classes to parameters
        """
        for i, j in zip(self.motifs, self.trials):
            cur_dict = {}
            cur_dict["class"] = "L" if j == 1 else "R"
            cur_dict["stim_name"] = i
            self.parameters["stims"][i] = self.parameters["stim_path"]+"/"+i
            self.parameters["block_design"]["blocks"]["default"]["conditions"].append(cur_dict)

    def session_post(self):
        """
        Closes out the sessions
        """
        self.parameters = copy.deepcopy(
            self.starting_params)  # copies the original conditions every new session
        self.get_conditions()  # get conditions for trial in block (e.g. ABA, BAB, BBA, ...)
        self.get_motifs()  # builds specific motif sequences (e.g. [A12, B1, A40])
        self.build_block()  # magp - build wav files from the generated sequences
        self.log.info('ending session')
        self.trial_q = None
        return None
