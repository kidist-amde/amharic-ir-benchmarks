import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from ColBERT_AM.parameters import DEVICE
from ColBERT_AM.modeling.colbert import AmharicColBERT
from ColBERT_AM.utils.utils import print_message, load_checkpoint


def load_model(args, do_print=True):
    colbert = AmharicColBERT.from_pretrained('./bert-medium-amharic',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint
