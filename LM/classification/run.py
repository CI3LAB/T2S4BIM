from arguments import get_args_parser, get_model_classes
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm, trange
import numpy as np
from collections import Counter
import random
from utils.data_processor import DataProcessor
from utils.data_utils import InputExample, InputFeature_pre, InputFeature, load_examples
from sklearn.metrics import f1_score, accuracy_score
import logging
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, AutoConfig
# from torch.utils.tensorboard import SummaryWriter
import json
logger = logging.getLogger(__name__)
# writer = SummaryWriter(log_dir=f"./runs")

def get_acc_f1(labels, predictions, idx2label, all_labels):
    assert len(labels) == len(predictions)
    accuracy = round(accuracy_score(labels, predictions) * 100, 2)

    labels_str = [idx2label[idx] for idx in labels]
    predictions_str = [idx2label[idx] for idx in predictions]

    accuracies = {i:0 for i in all_labels}
    tp = {i:0 for i in all_labels}
    fp = {i:0 for i in all_labels}
    fn = {i:0 for i in all_labels}

    for i in range(len(labels_str)):
        l = labels_str[i]
        p = predictions_str[i]
        if l == p:
            tp[l] += 1
        else:
            fn[l] += 1
            fp[p] += 1
    
    for i in all_labels:
        if tp[i] + fn[i] == 0:
            accuracies[i] = 0
        else:
            accuracies[i] = tp[i] / (tp[i] + fn[i])
        accuracies[i] = round(accuracies[i] * 100, 2)

    return accuracy, accuracies

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train(args, train_dataset, dev_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size
    train_sampler = RandomSampler(train_dataset)
    if "bert" in args.model_type:
        collate_fn = InputFeature_pre.collate_fct
    else:
        collate_fn = InputFeature.collate_fct
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  collate_fn=collate_fn
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
    # Check if continuing training from a checkpoint
    # if os.path.exists(args.model_name_or_path):
    #     # set global_step to gobal_step of last saved checkpoint from model path
    #     try:
    #         global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    #     except ValueError:
    #         global_step = 0
    #     epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    #     steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    #     logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    #     logger.info("  Continuing training from epoch %d", epochs_trained)
    #     logger.info("  Continuing training from global step %d", global_step)
    #     logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

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
            if "bert" in args.model_type:
                inputs = {
                    "input_ids":batch[0],
                    "token_type_ids":batch[1],
                    "attention_mask":batch[2],
                    "label_ids":batch[3],
                    "mode":"train"
                }
            else:
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "label_ids":batch[2],
                    "mode":"train"
                }
            
            loss= model(**inputs)

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
                        acc, acc_str = evaluate(args, dev_dataset, model, tokenizer)
                        # writer.add_scalar("eval_loss", eval_loss, logging_step)
                        # writer.add_scalar("macro_f1", macro_f1, logging_step)
                        logging_step += 1
                        if best_score < acc:
                            best_score = acc
                            output_dir = os.path.join(args.output_dir, "best_checkpoint")
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)

                            torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                            tokenizer.save_pretrained(output_dir)

                            torch.save(args, os.path.join(output_dir, "training_args.bin"))

                            logger.info("Saving model checkpoint to %s", output_dir)
            
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
        # evaluate after every epoch
        if args.evaluate_after_epoch:
            acc, acc_str = evaluate(args, dev_dataset, model, tokenizer)
            #output_eval_file = os.path.join(args.output_dir, "train_results.txt")
            #with open(output_eval_file, "a") as f:
            #    f.write('***** Predict Result for Dataset {} Seed {} *****\n'.format(args.data_dir, args.seed))
            #    f.write(result_str)
            #writer.add_scalar("eval_loss", eval_loss, epoch_num)
            #writer.add_scalar("trigger_f1", trigger_f1, epoch_num)

            if best_score < acc:
                best_score = acc
                output_dir = os.path.join(args.output_dir, "best_checkpoint")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                torch.save(model.state_dict(), os.path.join(output_dir, "model"))
                if "bert" in args.model_type:
                    tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))

                logger.info("Saving model checkpoint to %s", output_dir)

        epoch_num += 1

    return global_step, tr_loss / global_step

def evaluate(args, dev_dataset, model, tokenizer):
    # Eval!
    dev_sampler = SequentialSampler(dev_dataset)
    if "bert" in args.model_type:
        collate_fn = InputFeature_pre.collate_fct
    else:
        collate_fn = InputFeature.collate_fct
    dev_dataloader = DataLoader(dev_dataset,
                                 sampler=dev_sampler,
                                 batch_size=args.per_gpu_eval_batch_size,
                                 collate_fn=collate_fn
                                 )
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(dev_dataset))
    logger.info("  Batch size = %d", args.per_gpu_eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()

    preds = []
    trues = []
    for batch in tqdm(dev_dataloader, desc='Evaluating'):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            if "bert" in args.model_type:
                inputs = {
                    "input_ids":batch[0],
                    "token_type_ids":batch[1],
                    "attention_mask":batch[2],
                    "label_ids":batch[3],
                    "mode":"test"
                }
            else:
                inputs = {
                    "input_ids":batch[0],
                    "attention_mask":batch[1],
                    "label_ids":batch[2],
                    "mode":"test"
                }

            loss, logits, label_ids = model(**inputs)
            assert logits.size() == label_ids.size()
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            eval_loss += loss.item()
            nb_eval_steps += 1

            for i in range(len(logits)):
                preds.append(logits[i])
                trues.append(label_ids[i])

    assert len(preds) == len(trues)

    eval_loss /= nb_eval_steps

    print('eval_loss={}'.format(eval_loss))

    acc, accs = get_acc_f1(trues, preds, args.id2label, args.all_labels)

    print('[accuracy]')
    print('accuracy={:.4f}'.format(acc))

    metric = 'eval_loss={}\n'.format(eval_loss)
    metric += '[accuracy]\t{:.4f}\n'.format(acc)

    f1s_str = '[accuracy]\t{:.4f}\n'.format(acc)
    # f1s_str += 'For each categories:\n'
    # for i in args.all_labels:
    #     f1s_str += "accuracy of {} is: {}\n".format(i, accs[i])
    f1s_str += "\n"

    print(metric)
    return acc, f1s_str


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
    if "bert" in args.model_type:
        logger.info("Using pre-trained models...")
        model_classes = get_model_classes()
        model_config = model_classes[args.model_type]

        tokenizer = model_config['tokenizer'].from_pretrained(
            args.model_name_or_path
        )
    else:
        tokenizer = "no"

    logger.info("Loading dataset from run.py...")

    data_processor = DataProcessor(args.data_dir, args)

    args.label2id = data_processor.label2idx
    args.id2label = data_processor.idx2label
    labels = data_processor.all_labels
    args.num_labels = len(labels)
    args.all_labels = labels

    if args.model_type == "textcnn":
        from models.Textcnn import Textcnn
        model = Textcnn(args, data_processor)
    elif args.model_type == "textrnn":
        from models.Textrnn import Textrnn
        model = Textrnn(args, data_processor)
    elif args.model_type == "textrnn_att":
        from models.Textrnn_att import Textrnn_att
        model = Textrnn_att(args, data_processor)
    elif args.model_type == "textrcnn":
        from models.Textrcnn import Textrcnn
        model = Textrcnn(args, data_processor)
    elif args.model_type == "dpcnn":
        from models.Dpcnn import Dpcnn
        model = Dpcnn(args, data_processor)
    elif args.model_type == "bilstm":
        from models.BiLSTM import BiLSTM
        model = BiLSTM(args, data_processor)
    else: # "bert" in args.model_type
        assert "bert" in args.model_type
        from models.Bert import Bert
        model = Bert(args, tokenizer)
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
        global_step, tr_loss = train(args, train_dataset, dev_dataset, model, tokenizer)
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
        if "bert" in args.model_type:
            tokenizer = model_config['tokenizer'].from_pretrained(checkpoint)
        state_dict = torch.load(os.path.join(checkpoint, "model"))
        model.load_state_dict(state_dict)

        model.to(args.device)

        dev_dataset = load_examples(args, data_processor, 'test', tokenizer)
        acc, acc_str = evaluate(args, dev_dataset, model, tokenizer)
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "a") as f:
            f.write('***** Predict Result for Dataset {} Seed {} Type {} Ratio {} *****\n'.format(args.data_dir, args.seed, args.type, args.sample_ratio))
            f.write(acc_str)
    

if __name__ == "__main__":
    main()
