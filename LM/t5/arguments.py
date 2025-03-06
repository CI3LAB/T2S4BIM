import argparse
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_args_parser():

    parser = argparse.ArgumentParser(description="Command line interface for dialogue understanding.")
    
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data directory."
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or shortcut name of the model."
    )

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--max_seq_length", default=280, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--max_seq_length_label", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for label. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval.")
    parser.add_argument("--do_eval_topk", action="store_true", help="Whether to run eval in top-k manner.")

    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--evaluate_after_epoch",
        action="store_true",
        help="Whether to run evaluation after every epoch.",
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=3e-4, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--warmup_proportion", default=0, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--weight_decay", default=5e-5, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument("--logging_steps", type=str, default='0.1', help="Log every X updates steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    parser.add_argument("--dropout_prob", type=float, default=0.3, help="Dropout rate.")

    parser.add_argument("--do_sample", action="store_true", help="Whether to sample for training.")
    parser.add_argument("--sample_ratio", type=float, default=1.0, help="Sample ratio.")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args
    return args

def get_args():
    return _GLOBAL_ARGS
