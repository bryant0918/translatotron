# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:02:18 2022

@author: Bryant McArthur
"""

from sacrebleu.metrics import BLEU

bleu = BLEU()


def get_bleu(referencefile, translatedfile):
    with open(referencefile, 'r', encoding='utf-8') as rfile:
        refs = rfile.readlines()
        refs = [refs]

        with open(translatedfile, 'r', encoding='utf-8') as tfile:
            sys = tfile.readlines()
            sys = sys

            # print(bleu.corpus_score(sys, refs))

    return bleu.corpus_score(sys, refs)


if __name__ == "__main__":
    translated = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Demo\\00001-f000001_synth_asr.txt"
    reference = "C:\\Users\\bryan\\Documents\\School\\Winter 2023\\CS 601R\\Final Project\\Demo\\00001-f000001_ref.txt"

    bleu_score = get_bleu(reference, translated)
    print(bleu_score)

    pass



