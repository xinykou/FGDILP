_base_ = [
    '../_base_/models/others/dexperts-gpt2-l.py',
    '../_base_/datasets/rtp-test-nontoxic-8k.py',
    '../_base_/common.py',
]

batch_size = 48