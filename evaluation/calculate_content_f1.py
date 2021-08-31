import numpy as np
import argparse
summary_words = ['in', 'a', 'this', 'to', 'free', 'the', 'video', 'and', 'learn', 'from', 'on', 'with', 'how', 'tips', 'for', 'of', 'expert', 'an']


def cal_content_f1(file):
    F = open(file)
    data = F.readlines()
    F.close()
    items = []
    one_item = []
    for line in data:
        if line != '\n':
            one_item.append(line)
        else:
            items.append(one_item)
            one_item = []


    new_items = []
    for item in items:
        new_item = {}
        ref = item[2].split()
        gen = item[1].split()
        new_item['ref_text'] = ref
        new_item['hyp_text'] = gen
        remove_pair = []
        keep_pair = []
        ref_remove_words = []
        hyp_remove_words = []
        for line in item[4:]:
            numbers = [item for item in line.split('\t') if item != ''][:2]
            if ref[int(numbers[0].split(':')[0])] in summary_words and int(numbers[0].split(':')[1]) == 1:
                ref_remove_words.append(ref[int(numbers[0].split(':')[0])])
                hyp_remove_words.append(gen[int(numbers[1].split(':')[0])])
                remove_pair.append(numbers)
                continue
            else:
                keep_pair.append(numbers)
        new_item['remove_pair'] = remove_pair
        new_item['keep_pair'] = keep_pair
        new_item['ref_remove_words'] = ref_remove_words
        new_item['hyp_remove_words'] = hyp_remove_words

        ref_word_bag = []
        hyp_word_bag = []
        for word in new_item['ref_text']:
            if word not in ref_word_bag and word not in new_item['ref_remove_words']:
                ref_word_bag.append(word)

        for word in new_item['hyp_text']:
            if word not in hyp_word_bag and word not in new_item['hyp_remove_words']:
                hyp_word_bag.append(word)

        new_item['ref_word_bag'] = ref_word_bag
        new_item['hyp_word_bag'] = hyp_word_bag

        ref_align_number = sum([int(item[0].split(':')[1]) for item in keep_pair])
        hyp_align_number = sum([int(item[1].split(':')[1]) for item in keep_pair])
        ref_len  = len(ref_word_bag)
        hyp_len  = len(hyp_word_bag)

        recall = hyp_align_number / hyp_len
        precision = ref_align_number / ref_len
        f1 = 2 * (precision * recall) / (precision + recall)
        new_item['f1'] = f1
        new_items.append(new_item)

    print(np.mean([item['f1'] for item in new_items]))






if __name__ == "__main__":
    # init_logger('test_rouge.log')
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', type=str, default="./results/T5_text_only_align.txt-align.out",
                        help='alignment-file')

    args = parser.parse_args()

    cal_content_f1(args.a)
