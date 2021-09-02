from rouge import rouge_n_sentence_level
from rouge import rouge_l_sentence_level
from rouge import rouge_n_summary_level
from rouge import rouge_l_summary_level
from rouge import rouge_w_sentence_level
from rouge import rouge_w_summary_level
import numpy as np
import argparse

def calculate_rouge(summary_sentences, reference_sentences):
    rouge1 = []
    rouge2 = []
    rougel = []
    for i in range(len(reference_sentences)):
        score = rouge_n_sentence_level(summary_sentences[i], reference_sentences[i], 1)
        rouge1.append(score.f1_measure)
        _, _, rouge_2 = rouge_n_sentence_level(summary_sentences[i], reference_sentences[i], 2)
        rouge2.append(rouge_2)
        _, _, rouge_l = rouge_l_sentence_level(summary_sentences[i], reference_sentences[i])
        rougel.append(rouge_l)
    print(np.mean(rouge1))
    print(np.mean(rouge2))
    print(np.mean(rougel))

if __name__ == "__main__":
    # init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="./datasets/3layer_without_imgtransformer.txt",
                        help='candidate file')
    parser.add_argument('-r', type=str, default="./datasets/test.tgt",
                        help='reference file')
    args = parser.parse_args()

    cand_file = open(args.c)
    cand_data = cand_file.readlines()
    cand_data = [item.strip('\n').split() for item in cand_data]

    ref_file = open(args.r)
    ref_data = ref_file.readlines()
    ref_data = [item.strip('\n').split() for item in ref_data]

    calculate_rouge(cand_data, ref_data)