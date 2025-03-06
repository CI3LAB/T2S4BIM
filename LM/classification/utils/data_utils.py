import logging
from torch.utils.data._utils.collate import default_collate
import torch
from torch.utils.data import TensorDataset, Dataset
import re
import os

logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, sentence, label):
        self.guid = guid
        self.sentence = sentence
        self.label = label

class InputFeature_pre(object):
    def __init__(self, guid, input_ids, token_type_ids, attention_mask, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids
        
    @staticmethod
    def collate_fct(batch):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        batch_lens = torch.sum(batch_tuple[2], dim=-1, keepdim=False)
        max_len = batch_lens.max().item()
        results = ()
        for item in batch_tuple:
            if item.dim() >= 2:
                results += (item[:, :max_len],)
            else:
                results += (item,)
        return results

class InputFeature(object):
    def __init__(self, guid, input_ids, attention_mask, label_ids):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_ids = label_ids
        
    @staticmethod
    def collate_fct(batch):
        r'''
        This function is used to collate the input_features.
        Args:
            batch (:obj:`List[Union[Dict, InputFeatures]]`): A batch of the current data.
        Returns:
            :obj:`InputFeatures`: Return the :py:class:`~openprompt.data_utils.data_utils.InputFeatures of the current batch of data.
        '''
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
        max_len = batch_lens.max().item()
        results = ()
        for item in batch_tuple:
            if item.dim() >= 2:
                results += (item[:, :max_len],)
            else:
                results += (item,)
        return results


def convert_examples_to_features(
    examples,
    data_processor,
    max_seq_length
):
    word2id = data_processor.word2id

    features = []
    for example in examples:
        guid = example.guid
        # get tokens of each sentence
        token_ids = []
        words = example.sentence.split(' ')
        for word in words:
            if word in word2id:
                token_ids.append(word2id.index(word))
            else:
                token_ids.append(word2id.index('<UNK>'))
        # get label_ids of each sentence
        label_ids = data_processor.label2idx[example.label]

        seq_len = len(token_ids)
        if len(token_ids) < max_seq_length:
            input_mask = [1] * len(token_ids) + [0] * (max_seq_length - len(token_ids))
            token_ids += ([0] * (max_seq_length - len(token_ids))) # index of <PAD> is 0
        else:
            input_mask = [1] * max_seq_length
            token_ids = token_ids[:max_seq_length]
            seq_len = max_seq_length

        assert len(token_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
    
        features.append(
            InputFeature( guid=guid,
                          input_ids=token_ids,
                          attention_mask=input_mask,
                          label_ids=label_ids)
        )
    
    return features

def convert_examples_to_features_pre(
    examples,
    data_processor,
    max_seq_length,
    tokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    features = []
    for example in examples:
        guid = example.guid
        # get tokens of each sentence
        tokens = tokenizer.tokenize(example.sentence)
        # get label_ids of each sentence
        label_ids = data_processor.label2idx[example.label]
        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
        tokens += [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)
        # add cls token
        if cls_token_at_end:
            tokens += [cls_token]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
    
        features.append(
            InputFeature_pre( guid=guid,
                          input_ids=input_ids,
                          token_type_ids=segment_ids,
                          attention_mask=input_mask,
                          label_ids=label_ids)
        )
    
    return features

def load_examples(args, data_processor, split, tokenizer=None):
    logger.info("Loading and converting data from data_utils.py...")
    # Load data features from cache or dataset file
    if args.sample_ratio >= 0 and split == "train":
        examples = data_processor.get_examples_sample(sample_ratio=args.sample_ratio, seed=args.seed, split=split)
    else:
        examples = data_processor.get_examples(split=split)

    if "bert" in args.model_type:
        features = convert_examples_to_features_pre(
            examples,
            data_processor,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
        )
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_token_type_ids, all_attention_mask, all_label_ids)
    else: # for other vanilla machine learning models
        features = convert_examples_to_features(
            examples,
            data_processor,
            args.max_seq_length,
        )
        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_label_ids)
        
    return dataset