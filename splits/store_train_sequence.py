import numpy as np
import pdb


def read_split(txt_file):
    # read the txt file, sort the entries by seqence names
    # and store them as dictionaries
    file_dic = {}
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            seq_name = line.split(' ')[0]
            frm_lr = line.split(' ')[1]+line.split(' ')[2]
            if seq_name in file_dic:
                file_dic[seq_name].append(frm_lr)
            else:
                file_dic[seq_name] = [frm_lr]

    # separate the data to 4 parts as uniform as possible
    seq_names = []
    seq_lens = []
    len_full = 0
    for key in file_dic.keys():
        seq_names.append(key)
        seq_lens.append(len(file_dic[key]))
        len_full += len(file_dic[key])
    seq_lens = [i/len_full for i in seq_lens]
    seq_rank = np.argsort(seq_lens)[::-1].tolist()

    seq_names = [seq_names[i] for i in seq_rank]
    seq_lens = sorted(seq_lens, reverse=True)
    split_seqs = {}
    for i in range(3):
        seq_names_rest, seq_lens_rest = [], []
        split_seqs[str(i)] = [seq_names[0]]
        base = seq_lens[0]
        for j in range(1, len(seq_names)):
            if base + seq_lens[j] <= 0.25:
                base += seq_lens[j]
                split_seqs[str(i)].append(seq_names[j])
            else:
                seq_names_rest.append(seq_names[j])
                seq_lens_rest.append(seq_lens[j])
        print(base)
        seq_lens = seq_lens_rest
        seq_names = seq_names_rest

    split_seqs[str(3)] = seq_names_rest

    train_split = {'seq_splits': split_seqs, 'file_corres': file_dic}
    return train_split

def read_split_new(txt_file):
    # read the txt file, sort the entries by seqence names
    # and store them as dictionaries
    file_dic = {}
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            seq_name = line.split(' ')[0]
            frm_lr = line.split(' ')[1]+line.split(' ')[2]
            if seq_name in file_dic:
                file_dic[seq_name].append(frm_lr)
            else:
                file_dic[seq_name] = [frm_lr]

    # split the data by 1%, 5%, 10%, 20%, 30%, 50%
    sep_rates = [1, 5, 10, 20, 30, 50]
    seq_names_orig = []
    seq_lens_orig = []
    key_len = {}
    len_full = 0
    for key in file_dic.keys():
        seq_names_orig.append(key)
        seq_lens_orig.append(len(file_dic[key]))
        len_full += len(file_dic[key])
        key_len[key] = len(file_dic[key])
    seq_lens = [i/len_full for i in seq_lens_orig]
    seq_rank = np.argsort(seq_lens)[::-1].tolist()
    # pdb.set_trace()

    seq_names = [seq_names_orig[i] for i in seq_rank]
    seq_lens = sorted(seq_lens, reverse=True)
    split_seqs, split_seqs_rest, split_lens = {}, {}, {}
    bases = []
    for rate in sep_rates:
        split_seqs[str(rate)], split_seqs_rest[str(rate)], split_lens[str(rate)] = [], [], []
        base = 0
        for j in range(len(seq_names)):
            if base + seq_lens[j] <= rate/100:
                base += seq_lens[j]
                split_seqs[str(rate)].append(seq_names[j])
                split_lens[str(rate)].append(key_len[seq_names[j]])
            else:
                split_seqs_rest[str(rate)].append(seq_names[j])
        bases.append(base)

    train_split = {'seq_splits': split_seqs,
                   'seq_splits_rest': split_seqs_rest,
                   'file_corres': file_dic}
    return train_split

def get_depth_files(txt_file):
    file_list = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(' ')[0]
            seq_name = line.split('/')[0] + '/' + line.split('/')[1]
            frm = int(line.split('/')[4].split('.')[0])
            seq_name = seq_name + '_' + str(frm)
            file_list.append(seq_name)
    return file_list

def save_depth_txt(file_list, txt_file, savepath):
    store_lines = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            seq_name = line.split(' ')[0]
            frm = line.split(' ')[1]
            line_name = seq_name + '_' + frm
            if line_name in file_list:
                # store this line
                store_lines.append(line)
    with open(savepath, 'a') as f:
        for line in store_lines:
            f.write(line)
            f.write('\n')

def save_depth_eval_txt(txt_file, savepath):
    store_lines = []
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            line = line.split(' ')[0]
            seq_name = line.split('/')[0] + '/' + line.split('/')[1]
            frm = int(line.split('/')[4].split('.')[0])
            seq_name = seq_name + ' ' + str(frm) + ' ' + 'l'
            store_lines.append(seq_name)
    with open(savepath, 'a') as f:
        for line in store_lines:
            f.write(line)
            f.write('\n')
    pdb.set_trace()

if __name__ == "__main__":
    split_folder = 'eigen_zhou_depth'
    train_file = split_folder + '/train_files_bts.txt'
    train_split = read_split_new(train_file)
    np.save(split_folder+'/train_split_bts.npy', train_split)

    # file_list = get_depth_files(split_folder + '/eigen_train_files_with_gt.txt')
    # save_depth_txt(file_list, 'eigen_zhou/train_files.txt', 'eigen_zhou_depth/train_files.txt')

    # save_depth_eval_txt('eigen_zhou_depth/eigen_test_files_with_gt.txt', 'eigen_zhou_depth/val_files.txt')
    # save_depth_eval_txt('eigen_zhou_depth/eigen_train_files_with_gt.txt', 'eigen_zhou_depth/train_files_bts.txt')
