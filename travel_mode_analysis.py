import os
import logging
from glob import glob
import math
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm, trange
import numpy as np
from sklearn import metrics
import torch
from torch import nn
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from transformers import AdamW, get_linear_schedule_with_warmup

from config import FR_LTM_Config
from gridder import Gridder
from model import FR_LTM_ForTravelModeClassification
from util import (
        Preprocess4BiSTLM4TravelModeRecognition,
        LazyBiSTLMDataset4TravelModeRecognition,
        batch_list_to_batch_tensors
    )

def _get_max_epoch_model(output_dir):
    fn_model_list = glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None
    
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def gather_all_tensors(input_tensor):
    """Collect tensor data from all GPUs in a distributed environment"""
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, input_tensor)
    return torch.cat(gather_list, dim=0)

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default="", type=str,
                        help="The input data file name.")
    parser.add_argument("--eval_src_file", default=None, type=str,
                        help="The input data file name.")
    parser.add_argument("--test_src_file", default=None, type=str,
                        help="The input data file name.")
    parser.add_argument("--loss_file", default="loss.npy", type=str,
                        help="")
    parser.add_argument("--model_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--output_dir", default="./outputs_ft_tmr", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir", default='./log', type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--log_file", default="train_ft_tmr.log", type=str,
                        help="The output file where the log will be written.")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path", default=None, type=str,
                        help="The file of pretraining optimizer.")
    parser.add_argument("--config_path", default="model.json", type=str,
                        help="Pretrained config path if not the same as model_name")
    parser.add_argument("--gridder_name", default="", type=str,
                        help="Pretrained gridder name or path if not the same as model_name")
    
    # Other parameters
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help="max position embeddings")
    parser.add_argument('--intermediate_size', type=int, default=3072,
                        help="max position embeddings")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--test_batch_size", default=4, type=int,
                        help="Total batch size for test.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for hidden states.")
    parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
                        help="Dropout rate for attention probabilities.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local-rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--max_len_a', type=int, default=0,
                        help="Truncate_config: maximum length of segment A.")
    parser.add_argument('--max_len_b', type=int, default=0,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument("--mask_prob", default=0.20, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    
    parser.add_argument('--max_pred', type=int, default=20,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")

    parser.add_argument('--mask_source_words', action='store_true',
                        help="Whether to mask source words for training")
    parser.add_argument('--skipgram_prb', type=float, default=0.0,
                        help='prob of ngram mask')
    parser.add_argument('--skipgram_size', type=int, default=1,
                        help='the max size of ngram mask')
    
    parser.add_argument('--gridding_level', type=int, default=15,
                        help='the level of gridding level of google s2')
    parser.add_argument("--grid_file", default="./grid/cellID2gridID.json", type=str,
                        help="The grid file name.")
    parser.add_argument("--cid2cellId_file", default=None, type=str,
                        help="The cid2cellId file name.")
    parser.add_argument('--special_grid_num', type=int, default=5,
                        help='the number of special grids')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='')
    parser.add_argument('--num_hidden_layers', type=int, default=12,
                        help='')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help='')
    parser.add_argument('--type_grid_size', type=int, default=2,
                        help='')
    parser.add_argument("--is_hierarchical", action='store_true',
                        help="Whether to use is_hierarchical.")
    parser.add_argument("--recover", action='store_true', default=False,
                        help="Whether to recover.")
    
    parser.add_argument("--is_decoder", action='store_true', default=False,
                        help="")
    parser.add_argument("--add_cross_attention", action='store_true', default=False,
                        help="")
    parser.add_argument("--rotary_value", action='store_true', default=False,
                        help="")
    parser.add_argument("--output_hidden_states", action='store_true', default=False,
                        help="")
    parser.add_argument("--output_attentions", action='store_true', default=False,
                        help="")
    
    parser.add_argument('--num_label', type=int, default=6,
                        help='')
    parser.add_argument("--optim_path", default=None, type=str,
                        help="")
    parser.add_argument("--amp_path", default=None, type=str,
                        help="")
    parser.add_argument("--sche_path", default=None, type=str,
                        help="")
    
    parser.add_argument('--embedding_size', type=int, default=768,
                        help='')
    
    parser.add_argument('--scaling_factor', type=int, default=15,
                        help='')
    
    parser.add_argument("--truncateBySampling", action='store_true', default=False,
                        help="")
    parser.add_argument("--is_unidirectional_selfAttention", action='store_true', default=False,
                        help="")
    
    parser.add_argument("--gen_pos_idx", action='store_true', default=False,
                        help="")
    parser.add_argument('--k_minute_timeSlicing', type=int, default=-1,
                        help='')
    
    parser.add_argument("--traj_info", default="traj_info.json", type=str,
                        help="微调数据集信息")
    args = parser.parse_args()

    if not(args.model_recover_path and Path(args.model_recover_path).exists()):
        args.model_recover_path = None
        
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)

    logging.basicConfig(filename="%s/%s" % (args.log_dir, args.log_file), 
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        dist.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    
    gridder = Gridder(gridding_level=args.gridding_level, 
                      grid_file=args.grid_file)
    config = FR_LTM_Config.from_json_file(args.config_path)
    config.is_decoder=args.is_decoder
    config.add_cross_attention=args.add_cross_attention
    config.rotary_value=args.rotary_value
    config.scaling_factor = args.scaling_factor
    config.hidden_dropout_prob = args.hidden_dropout_prob
    config.attention_probs_dropout_prob = args.attention_probs_dropout_prob
    with open("%s/%s" % (args.data_dir, args.traj_info), "r", encoding="UTF-8") as file:
        traj_info = json.load(file)
    config.num_labels = traj_info["max_label_idx"] + 1
    args.num_label = traj_info["max_label_idx"] + 1
    config.to_json_file("%s/model.json" % (args.output_dir))
    
    if args.local_rank == 0:
        dist.barrier()
        logger.info(args.__dict__)
    
    if args.do_train:
        print("Loading Train Dataset", args.data_dir)
        bi_uni_pipeline = [Preprocess4BiSTLM4TravelModeRecognition(gridder.grid, args.max_seq_length,
                                                                   truncateBySampling=args.truncateBySampling,
                                                                   is_unidirectional_selfAttention=args.is_unidirectional_selfAttention)]
        file = os.path.join(
            args.data_dir, args.src_file)
        file = os.path.join(
            args.data_dir, args.src_file)
        if os.path.isfile(file):
            file = [file]
        elif os.path.isdir(file):
            file = glob("%s/*.json" % file)
        else:
            print("src_file 输入错误")
        real_num_train_epochs = args.num_train_epochs
        dummy_num_train_epochs = int(real_num_train_epochs * len(file))
        train_dataset = LazyBiSTLMDataset4TravelModeRecognition(
            file, args.train_batch_size, gridder, args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline,
            cur_epoch=0, gen_pos_idx=args.gen_pos_idx, k_minute_timeSlicing=args.k_minute_timeSlicing)
        if args.eval_src_file is not None:
            eval_file = os.path.join(args.data_dir, args.eval_src_file)
            if os.path.isfile(eval_file):
                eval_file = [eval_file]
            elif os.path.isdir(eval_file):
                eval_file = glob("%s/*.json" % eval_file)
            else:
                print("src_file 输入错误")
            eval_dataset = LazyBiSTLMDataset4TravelModeRecognition(
                eval_file, args.eval_batch_size, gridder, args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline,
                cur_epoch=0, gen_pos_idx=args.gen_pos_idx, k_minute_timeSlicing=args.k_minute_timeSlicing)
        if args.test_src_file is not None:
            test_file = os.path.join(args.data_dir, args.test_src_file)
            if os.path.isfile(test_file):
                test_file = [test_file]
            elif os.path.isdir(test_file):
                test_file = glob("%s/*.json" % test_file)
            else:
                print("src_file 输入错误")
            test_dataset = LazyBiSTLMDataset4TravelModeRecognition(
                test_file, args.test_batch_size, gridder, args.max_seq_length, bi_uni_pipeline=bi_uni_pipeline,
                cur_epoch=0, gen_pos_idx=args.gen_pos_idx, k_minute_timeSlicing=args.k_minute_timeSlicing)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
            _batch_size = args.train_batch_size
        else:
            train_sampler = DistributedSampler(train_dataset)
            _batch_size = args.train_batch_size // dist.get_world_size()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors, pin_memory=False)
        if args.eval_src_file is not None:
            if args.local_rank == -1:
                eval_sampler = RandomSampler(eval_dataset, replacement=False)
                _eval_batch_size = args.eval_batch_size
            else:
                eval_sampler = DistributedSampler(eval_dataset)
                _eval_batch_size = args.eval_batch_size // dist.get_world_size()
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=_eval_batch_size, sampler=eval_sampler,
                                                        num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors, pin_memory=False)
        if args.test_src_file is not None:
            if args.local_rank == -1:
                test_sampler = RandomSampler(test_dataset, replacement=False)
                _test_batch_size = args.test_batch_size
            else:
                test_sampler = DistributedSampler(test_dataset)
                _test_batch_size = args.test_batch_size // dist.get_world_size()
            test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=_test_batch_size, sampler=test_sampler,
                                                        num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors, pin_memory=False)
    t_total = int(len(train_dataloader) * dummy_num_train_epochs /
                  args.gradient_accumulation_steps)
    
    # Prepare model
    if args.recover:
        recover_step = _get_max_epoch_model(args.output_dir)
    else:
        recover_step = None

    if args.local_rank not in (-1, 0):
        # Make sure only the first process in distributed training will download model & vocab
        dist.barrier()
    global_step = 0

    model = FR_LTM_ForTravelModeClassification.from_pretrained(pretrained_model_path=args.model_path,
                                                        state_dict=torch.load(args.model_path, map_location='cpu'),
                                                        config=config)
    if args.local_rank == 0:
        dist.barrier()
    
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*t_total), num_training_steps=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
        
    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
                    args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    
    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        model.train()
        criterion = CrossEntropyLoss(ignore_index=-1)

        start_epoch = 1
        loss_list = []
        best_f1 = 0
        for i_epoch in trange(start_epoch, dummy_num_train_epochs+1, desc="Epoch", disable=args.local_rank not in (-1, 0)):
            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)',
                            disable=args.local_rank not in (-1, 0))
            for step, batch in enumerate(iter_bar):
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                input_ids, segment_ids, input_mask, position_ids, uid, labels = batch
                labels = labels.long()
                logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                               position_ids=position_ids,
                               attention_mask=input_mask,
                               output_attentions=args.output_attentions,
                               output_hidden_states=args.output_hidden_states)
                
                loss = criterion(logits.view(-1, args.num_label), labels.view(-1))

                loss_list.append(loss.item())
                if args.local_rank != -1:
                    if (step + 1) % 100 == 0:
                        np.save("%s/gpu%d_%s" % (args.output_dir, torch.distributed.get_rank(), args.loss_file), loss_list)

                if n_gpu > 1:    # mean() to average on multi-gpu.
                    loss = loss.mean()

                iter_bar.set_description(
                    'Iter (loss=%5.3f)' % (loss.item()))
            
                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizer), args.max_grad_norm)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm)

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
            
            if args.local_rank != -1:
                np.save("%s/gpu%d_%s" % (args.output_dir, torch.distributed.get_rank(), args.loss_file), loss_list)

            if args.eval_src_file is not None:
                model.eval()
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                local_losses_list, local_trues_list, local_preds_list = [], [], []
                with torch.no_grad():
                    for batch in tqdm(eval_dataloader):
                        batch = [
                                    t.to(device) if t is not None else None for t in batch]
                        input_ids, segment_ids, input_mask, position_ids, uid, labels = batch
                        labels = labels.long()
                        logits = model(input_ids=input_ids, token_type_ids=segment_ids,
                                    position_ids=position_ids,
                                    attention_mask=input_mask,
                                    output_attentions=args.output_attentions,
                                    output_hidden_states=args.output_hidden_states)
                        
                        loss = criterion(logits.view(-1, args.num_label), labels.view(-1))

                        labels = labels.view(-1)

                        logits = logits.view(-1, args.num_label)
                        pred = logits.argmax(dim=-1)

                        local_losses_list.append(loss)
                        local_trues_list.append(labels)
                        local_preds_list.append(pred)

                    local_losses_list = torch.tensor(local_losses_list, device=torch.distributed.get_rank())
                    local_trues_list = torch.cat(local_trues_list, dim=0)
                    local_preds_list = torch.cat(local_preds_list, dim=0)

                    global_losses = gather_all_tensors(local_losses_list)
                    global_trues = gather_all_tensors(local_trues_list)
                    global_preds = gather_all_tensors(local_preds_list)

                if dist.get_rank() == 0:
                    global_losses = global_losses.cpu().numpy()
                    global_trues = global_trues.cpu().numpy()
                    global_preds = global_preds.cpu().numpy()

                    global_preds = global_preds[global_trues >= 0]
                    global_trues = global_trues[global_trues >= 0]

                    dev_loss = np.mean(global_losses)
                    dev_acc = metrics.accuracy_score(global_trues, global_preds)
                    dev_metrics = metrics.classification_report(global_trues, global_preds, output_dict=True, digits=4)
                    logger.info("eval loss: %.5f, eval acc: %.5f" % (dev_loss, dev_acc))
                    logger.info(dev_metrics)

                    dev_f1 = dev_metrics["macro avg"]["f1-score"]
                
                    if dev_f1 > best_f1:
                        best_f1 = dev_f1
                        logger.info(
                            "** ** * Saving fine-tuned model and optimizer ** ** * ")
                        model_to_save = model.module if hasattr(
                            model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(
                            args.output_dir, "model.{0}.bin".format(i_epoch))
                        torch.save(model_to_save.state_dict(), output_model_file)
                        output_optim_file = os.path.join(
                            args.output_dir, "optim.{0}.bin".format(i_epoch))
                        torch.save(optimizer.state_dict(), output_optim_file)
                        if args.fp16:
                            output_amp_file = os.path.join(
                                args.output_dir, "amp.{0}.bin".format(i_epoch))
                            torch.save(amp.state_dict(), output_amp_file)
                        output_sched_file = os.path.join(
                            args.output_dir, "sched.{0}.bin".format(i_epoch))
                        torch.save(scheduler.state_dict(), output_sched_file)

                        logger.info("***** CUDA.empty_cache() *****")
                        torch.cuda.empty_cache()
            
            model.train()

if __name__ == "__main__":
    main()