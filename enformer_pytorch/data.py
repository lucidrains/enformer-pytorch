import torch
import torch.nn.functional as F

def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def str_to_seq_indices(seq_strs, padding = '.'):
    seq_strs = cast_list(seq_strs)
    char_to_index_map = {'a': 0, 'c': 1, 'g': 2, 't': 3, 'n': 4, padding: -1}
    seq_strs = map(lambda x: x.lower(), seq_strs)
    seq_indices = list(map(lambda seq_str: torch.Tensor(list(map(lambda char: char_to_index_map[char], seq_str))), seq_strs))
    return torch.stack(seq_indices).long()

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

# processing bed files

import pandas as pd
from random import randrange
from pathlib import Path
from pyfaidx import Fasta
from torch.utils.data import Dataset

class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        bed_file,
        fasta_file,
        context_length = None,
        return_seq_indices = False,
        filter_df_fn = identity,
        shift_augs = None,
        chr_bed_to_fasta_map = dict()
    ):
        super().__init__()
        bed_path = Path(bed_file)
        fasta_file = Path(fasta_file)

        assert bed_path.exists(), 'path to .bed file must exist'
        assert fasta_file.exists(), 'path to fasta file must exist'

        df = pd.read_csv(str(bed_path), sep = '\t', header = None)
        df = filter_df_fn(df)

        self.df = df
        self.seqs = Fasta(str(fasta_file))
        self.context_length = context_length
        self.return_seq_indices = return_seq_indices

        if exists(shift_augs):
            assert len(shift_augs) == 2, 'shift augs needs to be a tuple of 2, indicating min and max relative shifts inclusive - ex. (-2, 2) for [-2, -1, 0, 1, 2]'

        self.shift_augs = shift_augs

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        interval = self.df.iloc[ind]
        chr_name, start, end = (interval[0], interval[1], interval[2])
        interval_length = end - start

        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)

        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = max(start + min_shift, 0) - start
            max_shift = min(end + max_shift, chromosome_length) - end

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

            if start < 0:
                left_padding = -start
                start = 0

            if end > chromosome_length:
                right_padding = end - chromosome_length
                end = chromosome_length

        seq = ('.' * left_padding) + str(chromosome[start:end]) + ('.' * right_padding)
        seq_indices = str_to_seq_indices(seq)

        if self.return_seq_indices:
            return seq_indices.squeeze(0)

        seq_onehot = seq_indices_to_one_hot(seq_indices)
        return seq_onehot.squeeze(0)
