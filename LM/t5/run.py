from arguments import get_args_parser
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
from utils.utils import Utils
from utils.data_processor import DataProcessor
from utils.data_utils import InputExample, InputFeature, load_examples
from sklearn.metrics import f1_score, accuracy_score
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration
from tokenizers import AddedToken
# from torch.utils.tensorboard import SummaryWriter
import json
import re
logger = logging.getLogger(__name__)
# writer = SummaryWriter(log_dir=f"./runs")

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args, train_dataset, dev_dataset, model, tokenizer, utils):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size
                                  )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    args.logging_steps = eval(args.logging_steps)
    if isinstance(args.logging_steps, float):
        args.logging_steps = int(args.logging_steps * len(train_dataloader)) // args.gradient_accumulation_steps
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
        "lr": args.learning_rate},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": args.learning_rate},
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.warmup_proportion != 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Random seed = %d", args.seed)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    steps_trained_in_current_epoch = 0
    logging_step = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproductibility
    epoch_num = 1
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "labels":batch[2]
            }
            
            loss = model(**inputs).loss

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # writer.add_scalar("loss", loss.item(), global_step)
                # writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.evaluate_during_training:
                        intent_acc, element_acc, slot_f1, all_acc, acc_scores, f1s_str = evaluate(args, dev_dataset, model, tokenizer, utils)
                        print(f1s_str)
                        # writer.add_scalar("eval_loss", eval_loss, logging_step)
                        # writer.add_scalar("macro_f1", macro_f1, logging_step)
                        logging_step += 1
                        if best_score < slot_f1:
                            best_score = slot_f1
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))

                            logger.info("Saving model checkpoint to %s", output_dir)
        
        # evaluate after every epoch
        if args.evaluate_after_epoch:
            intent_acc, element_acc, slot_f1, all_acc, acc_scores, f1s_str = evaluate(args, dev_dataset, model, tokenizer, utils)
            print(f1s_str)
            #output_eval_file = os.path.join(args.output_dir, "train_results.txt")
            #with open(output_eval_file, "a") as f:
            #    f.write('***** Predict Result for Dataset {} Seed {} *****\n'.format(args.data_dir, args.seed))
            #    f.write(result_str)
            #writer.add_scalar("eval_loss", eval_loss, epoch_num)
            #writer.add_scalar("trigger_f1", trigger_f1, epoch_num)

            if best_score < slot_f1:
                best_score = slot_f1
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Saving model checkpoint to %s", output_dir)

        epoch_num += 1

    return global_step, tr_loss / global_step

def evaluate(args, dev_dataset, model, tokenizer, utils, out=False):
    # Eval!
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset,
                                 sampler=dev_sampler,
                                 batch_size=args.per_gpu_eval_batch_size
                                 )
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)

    model.eval()

    preds_str = []
    trues_str = []
    for batch in tqdm(dev_dataloader, desc='Evaluating'):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids":batch[0],
                "attention_mask":batch[1],
                "do_sample":False,
                "max_length":args.max_seq_length_label
            }
            
            output_sequences = model.generate(**inputs)

            for seq in output_sequences:
                str = tokenizer.decode(seq, skip_special_tokens=True)
                str = re.sub(r" +", " ", str)
                str = str.replace(" \n ", "\n")
                str = str.replace("\n ", "\n")
                str = str.replace(" \n", "\n")
                preds_str.append(str)
            for seq in batch[2]:
                seq[seq == -100] = 0
                str = tokenizer.decode(seq, skip_special_tokens=True)
                str = re.sub(r" +", " ", str)
                str = str.replace(" \n ", "\n")
                str = str.replace("\n ", "\n")
                str = str.replace(" \n", "\n")
                trues_str.append(str)
    
    # if out:
    # with open("prediction_mix_architect_{}.json".format(args.seed), 'w', encoding='UTF-8') as f:
    #     json.dump(preds_str, f, indent=4, ensure_ascii=False)

    assert len(preds_str) == len(trues_str)
    # try:
    intent_acc, element_acc, slot_f1, all_acc, acc_scores = utils.metric(trues_str, preds_str)
    # except:
    #     intent_acc, element_acc, slot_f1, all_acc = 0.0, 0.0, 0.0, 0.0
    #     acc_scores = {"creation":0, "deletion":0, "modification":0, "retrieval":0}

    f1s_str = '[intent accuaracy]\t{:.2f}\n'.format(intent_acc)
    f1s_str += '[element accuracy]\t{:.2f}\n'.format(element_acc)
    f1s_str += '[slot f1]\t{:.2f}\n'.format(slot_f1)
    f1s_str += '[all accuracy]\t{:.2f}\n'.format(all_acc)
    for cat in acc_scores:
        f1s_str += '[category {} accuracy]\t{:.2f}\n'.format(cat, acc_scores[cat])

    return intent_acc, element_acc, slot_f1, all_acc, acc_scores, f1s_str


def main():
    args = get_args_parser()

    args.device = torch.device("cuda")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    set_seed(args)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(AddedToken("\n", normalized=False))

    logger.info("Loading dataset from run.py...")

    data_processor = DataProcessor(args.data_dir, args)
    utils = Utils()

    model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = model.cuda()
    # for param in model.named_parameters():
    #     if 'label_embedding' in param[0]:
    #         print(param)
    # exit()
    
    # Training
    if args.do_train:
        train_dataset = load_examples(args, data_processor, 'train', tokenizer)
        # for evalute during training
        dev_dataset = load_examples(args, data_processor, 'dev', tokenizer)
        global_step, tr_loss = train(args, train_dataset, dev_dataset, model, tokenizer, utils)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    # if args.do_train:
    #     # Create output directory if needed
    #     if not os.path.exists(args.output_dir):
    #         os.makedirs(args.output_dir)

    #     logger.info("Saving model checkpoint to %s", args.output_dir)
    #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    #     # They can then be reloaded using `from_pretrained()`
    #     model_to_save = (
    #         model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
    #     model_to_save.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
        
    #     # Good practice: save your training arguments together with the trained model
    #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    #     output_dir = os.path.join(args.output_dir, "last_checkpoint")
    #     if not os.path.exists(output_dir):
    #        os.makedirs(output_dir)

    #     torch.save(model.state_dict(), os.path.join(output_dir, "model"))
    #     tokenizer.save_pretrained(output_dir)

    #     torch.save(args, os.path.join(output_dir, "training_args.bin"))
        
    #     logger.info("Saving model checkpoint to %s", output_dir)
    
    # Evaluation
    if args.do_eval:
        checkpoint = os.path.join(args.output_dir, 'best_checkpoint')
        # tokenizer = T5Tokenizer.from_pretrained(checkpoint)
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.add_tokens(AddedToken("\n", normalized=False))
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)

        model.to(args.device)

        dev_dataset = load_examples(args, data_processor, 'test', tokenizer)
        intent_acc, element_acc, slot_f1, all_acc, acc_scores, f1s_str = evaluate(args, dev_dataset, model, tokenizer, utils)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        print(f1s_str)
        with open(output_eval_file, "a") as f:
            f.write('***** Predict Result for Dataset {} Seed {} *****\n'.format(args.data_dir, args.seed))
            f.write(f1s_str)
            f.write('\n')
    

if __name__ == "__main__":
    main()
