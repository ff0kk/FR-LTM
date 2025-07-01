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
import torch
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AdamW, get_linear_schedule_with_warmup
import deepspeed
from torch.cuda.amp import autocast
from deepspeed.ops.adam import FusedAdam
from collections import OrderedDict

from config import FR_LTM_Config
from gridder import Gridder
from model import FR_LTM_ForCausalLM
from util import (
        Preprocess4Left2Right_DefaultPositionIds,
        LazyBiSTLMDataset4MLM_DefaultPosIds,
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

def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="./data", type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--src_file", default="trajs.json", type=str,
                        help="The input data file name.")
    parser.add_argument("--eval_src_file", default=None, type=str,
                        help="The input data file name.")
    parser.add_argument("--output_dir", default="./outputs", type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--eval_loss_saveDirName", default="eval_loss_acc", type=str, required=True,
                        help="")
    parser.add_argument("--log_dir", default='./log', type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--log_file", default="train.log", type=str,
                        help="The output file where the log will be written.")
    parser.add_argument("--loss_file", default="loss.npy", type=str,
                        help="")
    parser.add_argument("--acc_mlm_file", default="acc_mlm.npy", type=str,
                        help="")
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--optim_recover_path", default=None, type=str,
                        help="The file of pretraining optimizer.")
    parser.add_argument("--config_path", default="model.json", type=str,
                        help="Pretrained config path if not the same as model_name")
    
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
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float,
                        help="Beta1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float,
                        help="Beta2 for Adam optimizer.")
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
    parser.add_argument("--local_rank", type=int, default=-1,
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
    parser.add_argument("--mask_prob", default=0.20, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    
    parser.add_argument('--max_pred', type=int, default=20,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Number of workers for the data loader.")
    
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
    parser.add_argument('--embedding_size', type=int, default=768,
                        help='')
    parser.add_argument('--num_hidden_layers', type=int, default=12,
                        help='')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help='')
    parser.add_argument('--type_grid_size', type=int, default=2,
                        help='')
    parser.add_argument("--recover", action='store_true', default=False,
                        help="Whether to recover.")
    parser.add_argument("--save_per_real_epoch", action='store_true', default=False,
                        help="")

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
    
    parser.add_argument('--scaling_factor', type=int, default=15,
                        help='')
    parser.add_argument('--k_minute_timeSlicing', type=int, default=15,
                        help='')
    parser.add_argument("--ignore_unk", action='store_true', default=False,
                        help="")
    
    parser.add_argument("--save_per_specified_file_num", action='store_true', default=False,
                        help="")
    parser.add_argument('--specified_file_num', type=int, default=10,
                        help='')
    
    parser.add_argument('--zero_stage', type=int, default=1,
                        help='')
    parser.add_argument('--bf16', action='store_true',
                        help="")
    parser.add_argument('--min_loss_scale', type=int, default=1,
                        help='')
    
    parser.add_argument('--preNorm', action='store_true',
                        help="")
    
    parser.add_argument("--replace_static_dict_key", action='store_true',
                        help="")

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    if not(args.model_recover_path and Path(args.model_recover_path).exists()):
        args.model_recover_path = None

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("%s/%s" % (args.output_dir, args.eval_loss_saveDirName), exist_ok=True)
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
        deepspeed.init_distributed()
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))


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
    config = FR_LTM_Config(
            grid_num=len(gridder.cellID2gridID),
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            num_attention_heads=args.num_attention_heads,
            max_position_embeddings=args.max_position_embeddings,
            intermediate_size=args.intermediate_size,
            is_decoder=args.is_decoder,
            add_cross_attention=args.add_cross_attention,
            rotary_value=args.rotary_value,
            scaling_factor=args.scaling_factor
        )
    config.to_json_file("%s/%s" % (args.output_dir, args.config_path))

    if args.local_rank == 0:
        dist.barrier()
    
    if args.do_train:
        print("Loading Train Dataset", args.data_dir)
        bi_uni_pipeline = [Preprocess4Left2Right_DefaultPositionIds(indexer=gridder.grid, max_len=args.max_seq_length,
                                                                    k_minute_timeSlicing=args.k_minute_timeSlicing,
                                                                    ignore_unk=args.ignore_unk)]
        file = os.path.join(
            args.data_dir, args.src_file)
        if os.path.isfile(file):
            file = [file]
        elif os.path.isdir(file):
            file = glob("%s/*.json" % file)
        elif "*" in file:
            file = glob("%s/*.json" % file)
        else:
            print("src_file Input Error")
        real_num_train_epochs = args.num_train_epochs
        dummy_num_train_epochs = int(real_num_train_epochs * len(file))
        train_dataset = LazyBiSTLMDataset4MLM_DefaultPosIds(
            file_paths=file, 
            batch_size=int(args.train_batch_size / args.gradient_accumulation_steps), 
            max_len=args.max_seq_length, 
            bi_uni_pipeline=bi_uni_pipeline, cur_epoch=0)
        if args.eval_src_file is not None:
            eval_file = os.path.join(args.data_dir, args.eval_src_file)
            if os.path.isfile(eval_file):
                eval_file = [eval_file]
            elif os.path.isdir(eval_file):
                eval_file = glob("%s/*.json" % eval_file)
            else:
                print("src_file Input Error")
            eval_dataset = LazyBiSTLMDataset4MLM_DefaultPosIds(
                file_paths=eval_file, batch_size=args.eval_batch_size, max_len=args.max_seq_length, 
                bi_uni_pipeline=bi_uni_pipeline, cur_epoch=0)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
            _batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
            if args.eval_src_file is not None:
                eval_sampler = RandomSampler(eval_dataset, replacement=False)
                _eval_batch_size = args.eval_batch_size
        else:
            train_sampler = DistributedSampler(train_dataset)
            _batch_size = int(args.train_batch_size / args.gradient_accumulation_steps) // dist.get_world_size()
            if args.eval_src_file is not None:
                eval_sampler = DistributedSampler(eval_dataset)
                _eval_batch_size = args.eval_batch_size // dist.get_world_size()
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                       num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors, pin_memory=False)
        if args.eval_src_file is not None:
            eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=_eval_batch_size, sampler=eval_sampler,
                                                        num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors, pin_memory=False)
               
    t_total = int(len(train_dataloader) * dummy_num_train_epochs /
                  args.gradient_accumulation_steps)

    # Prepare model
    if args.recover:
        recover_step = _get_max_epoch_model(args.output_dir)
    else:
        recover_step = None
    
    global_step = 0
    if (recover_step is None) and (args.model_recover_path is None):
        model = FR_LTM_ForCausalLM(config)
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_weights = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)), map_location='cpu')
            # recover_step == number of epochs
            # global_step = math.floor(
            #     recover_step * t_total / args.num_train_epochs)
            global_step = math.floor(
                recover_step * t_total / dummy_num_train_epochs)
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_weights = torch.load(
                args.model_recover_path, map_location='cpu')
            
        if args.replace_static_dict_key:
            model = FR_LTM_ForCausalLM(config)
            new_state_dict = OrderedDict()
            for k, v in model_weights.items():
                new_k = k.replace("module.", "")
                new_state_dict[new_k] = v
            model.load_state_dict(new_state_dict)
        else:
            model = FR_LTM_ForCausalLM.from_pretrained(pretrained_model_path=args.model_recover_path,
                                                    state_dict=model_weights, config=config)
    
    if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        param_num_info = get_parameter_number(model)
        logger.info("** model parameter number info ** Total: %d; Trainable: %d" % (param_num_info["Total"],
                                                                                    param_num_info["Trainable"]))

    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, 
                          betas=(args.adam_beta1, args.adam_beta2),
                          eps=args.adam_epsilon,
                          weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion*t_total), num_training_steps=t_total)
        
    if args.local_rank != -1:
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("DistributedDataParallel")
        model = DDP(model, device_ids=[
                    args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        
    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        if os.path.exists(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step))):
            optim_recover = torch.load(os.path.join(
                args.output_dir, "optim.{0}.bin".format(recover_step)), map_location='cpu')
            if hasattr(optim_recover, 'state_dict'):
                optim_recover = optim_recover.state_dict()
            optimizer.load_state_dict(optim_recover)

        logger.info("***** Recover scheduler: %d *****", recover_step)
        if os.path.exists(os.path.join(
            args.output_dir, "sched.{0}.bin".format(recover_step))):
            scheduler_recover = torch.load(os.path.join(
                args.output_dir, "sched.{0}.bin".format(recover_step)), map_location='cpu')
            scheduler.load_state_dict(scheduler_recover)

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()
    
    ds_config = {
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,
        "zero_optimization": {
            "stage": args.zero_stage
        }
    }
    if args.fp16:
        ds_config["fp16"] = {
            "enabled": True,
            "min_loss_scale": args.min_loss_scale
        }
    else:
        ds_config["fp16"] = {
            "enabled": False
        }
    if args.bf16:
        ds_config["bf16"] = {
            "enabled": True
        }
    else:
        ds_config["bf16"] = {
            "enabled": False
        }

    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=optimizer_grouped_parameters,
        config=ds_config,
        optimizer=optimizer,
    )

    for i, g in enumerate(optimizer.param_groups):
        logger.info(f"group {i}, wd={g['weight_decay']}, #params={len(g['params'])}")


    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)

        model.train()
        if recover_step:
            start_epoch = recover_step+1
        else:
            start_epoch = 1
        loss_list = []
        acc_mlm_list = []
        eval_loss_list, eval_acc_mlm_list = [], [] 
        for i_epoch in trange(start_epoch, dummy_num_train_epochs+1, desc="Epoch", disable=args.local_rank not in (-1, 0)):
            if args.local_rank != -1:
                train_sampler.set_epoch(i_epoch)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)',
                            disable=args.local_rank not in (-1, 0))
            for step, batch in enumerate(iter_bar):
                batch = [
                    t.to(device) if t is not None else None for t in batch]
                input_ids, segment_ids, input_mask, lm_label_ids, position_ids = batch
                bilm_loss, correct_mlm = model(input_ids=input_ids, token_type_ids=segment_ids,
                                            position_ids=position_ids,
                                            attention_mask=input_mask,
                                            output_attentions=args.output_attentions,
                                            output_hidden_states=args.output_hidden_states,
                                            labels=lm_label_ids)
                loss = bilm_loss
                lm_label_ids = lm_label_ids.view(-1)
                lm_label_ids = lm_label_ids[lm_label_ids > 0]
                correct_mlm = correct_mlm / len(lm_label_ids)

                # logging for each step
                if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                    iter_bar.set_description(
                        'Iter (loss=%5.3f, acc_mlm=%5.3f)' % (loss.item(), correct_mlm.item()))
                loss_list.append(loss.item())
                acc_mlm_list.append(correct_mlm.item())
                if args.local_rank != -1:
                    if (step + 1) % 100 == 0:
                        np.save("%s/gpu%d_%s" % (args.output_dir, torch.distributed.get_rank(), args.loss_file), loss_list)
                        np.save("%s/gpu%d_%s" % (args.output_dir, torch.distributed.get_rank(), args.acc_mlm_file), acc_mlm_list)
                
                if (args.local_rank == -1 or torch.distributed.get_rank() == 0) \
                    and (step + 1) % 100 == 0:
                    logger.info("loss=%5.3f, acc_mlm=%5.3f" % (loss.item(), correct_mlm.item()))

                model.backward(loss)
                model.step()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    global_step += 1
                    scheduler.step()
            
            if args.local_rank != -1:
                np.save("%s/gpu%d_%s" % (args.output_dir, torch.distributed.get_rank(), args.loss_file), loss_list)
                np.save("%s/gpu%d_%s" % (args.output_dir, torch.distributed.get_rank(), args.acc_mlm_file), acc_mlm_list)


            # eval
            if args.eval_src_file is not None:
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.local_rank != -1:
                    eval_sampler.set_epoch(i_epoch)
                eval_iter_bar = tqdm(eval_dataloader, desc='Iter (loss=X.XXX)',
                                disable=args.local_rank not in (-1, 0))
                with torch.no_grad():
                    logger.info("***** eval *****")
                    total_correct_mlm = 0
                    total_mlm = 0
                    total_eval_loss = 0
                    eval_step = 0
                    
                    for _, batch in enumerate(eval_iter_bar):
                        batch = [
                            t.to(device) if t is not None else None for t in batch]
                        input_ids, segment_ids, input_mask, lm_label_ids, position_ids = batch
                        bilm_loss, correct_mlm = model(input_ids=input_ids, token_type_ids=segment_ids,
                                                    position_ids=position_ids,
                                                    attention_mask=input_mask,
                                                    output_attentions=args.output_attentions,
                                                    output_hidden_states=args.output_hidden_states,
                                                    labels=lm_label_ids)
                        loss = bilm_loss

                        total_correct_mlm += correct_mlm
                        total_eval_loss += loss
                        eval_step += 1
                        lm_label_ids = lm_label_ids.view(-1)
                        lm_label_ids = lm_label_ids[lm_label_ids > 0]
                        correct_mlm = correct_mlm / len(lm_label_ids)
                        total_mlm += len(lm_label_ids)

                        eval_iter_bar.set_description(
                            'Iter (loss=%5.3f, acc_mlm=%5.3f)' % (loss.item(), correct_mlm.item()))
                        
                    total_correct_mlm_ratio = total_correct_mlm / total_mlm
                    total_eval_loss = total_eval_loss / eval_step
                    eval_loss_list.append(total_eval_loss.item())
                    eval_acc_mlm_list.append(total_correct_mlm_ratio.item())
                    logger.info("***** epoch: %d, eval: avg_loss=%5.3f, avg_acc_mlm=%5.3f *****" % (i_epoch, total_eval_loss.item(), total_correct_mlm_ratio.item()))
                    if args.local_rank != -1:
                        np.save("%s/%s/gpu%d_%s" % (args.output_dir, args.eval_loss_saveDirName, torch.distributed.get_rank(), args.loss_file), eval_loss_list)
                        np.save("%s/%s/gpu%d_%s" % (args.output_dir, args.eval_loss_saveDirName, torch.distributed.get_rank(), args.acc_mlm_file), eval_acc_mlm_list)

            # Save a trained model
            if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
                if args.save_per_real_epoch: # Save the model after going through the data
                    if i_epoch % len(file) == 0:
                        logger.info(
                            "** ** * Saving fine-tuned model and optimizer ** ** * ")
                        model_to_save = strip_module_prefix(model.state_dict())
                        output_model_file = os.path.join(
                            args.output_dir, "model.{0}.bin".format(i_epoch))
                        torch.save(model_to_save, output_model_file)
                        output_optim_file = os.path.join(
                            args.output_dir, "optim.{0}.bin".format(i_epoch))
                        torch.save(optimizer.state_dict(), output_optim_file)
                        output_sched_file = os.path.join(
                            args.output_dir, "sched.{0}.bin".format(i_epoch))
                        torch.save(scheduler.state_dict(), output_sched_file)
                    else:
                        pass
                elif args.save_per_specified_file_num:
                    if i_epoch % args.specified_file_num == 0:
                        logger.info(
                            "** ** * Saving fine-tuned model and optimizer ** ** * ")
                        model_to_save = strip_module_prefix(model.state_dict())
                        output_model_file = os.path.join(
                            args.output_dir, "model.{0}.bin".format(i_epoch))
                        torch.save(model_to_save, output_model_file)
                        output_optim_file = os.path.join(
                            args.output_dir, "optim.{0}.bin".format(i_epoch))
                        torch.save(optimizer.state_dict(), output_optim_file)
                        output_sched_file = os.path.join(
                            args.output_dir, "sched.{0}.bin".format(i_epoch))
                        torch.save(scheduler.state_dict(), output_sched_file)
                    else:
                        pass
                else: # Save the model once per file
                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()
                    logger.info(
                        "** ** * Saving fine-tuned model and optimizer ** ** * ")
                    model_to_save = strip_module_prefix(model.state_dict())
                    output_model_file = os.path.join(
                        args.output_dir, "model.{0}.bin".format(i_epoch))
                    torch.save(model_to_save, output_model_file)
                    output_optim_file = os.path.join(
                        args.output_dir, "optim.{0}.bin".format(i_epoch))
                    torch.save(optimizer.state_dict(), output_optim_file)
                    output_sched_file = os.path.join(
                        args.output_dir, "sched.{0}.bin".format(i_epoch))
                    torch.save(scheduler.state_dict(), output_sched_file)

                    logger.info("***** CUDA.empty_cache() *****")
                    torch.cuda.empty_cache()

            if i_epoch < int(dummy_num_train_epochs):
                print("Loading Train Dataset", args.data_dir)
                bi_uni_pipeline = [Preprocess4Left2Right_DefaultPositionIds(indexer=gridder.grid, max_len=args.max_seq_length,
                                                                    k_minute_timeSlicing=args.k_minute_timeSlicing,
                                                                    ignore_unk=args.ignore_unk)]
                train_dataset = LazyBiSTLMDataset4MLM_DefaultPosIds(
                    file_paths=file, batch_size=int(args.train_batch_size / args.gradient_accumulation_steps),
                    max_len=args.max_seq_length, 
                    bi_uni_pipeline=bi_uni_pipeline, cur_epoch=i_epoch)
                if args.local_rank == -1:
                    train_sampler = RandomSampler(train_dataset, replacement=False)
                    _batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
                else:
                    train_sampler = DistributedSampler(train_dataset)
                    _batch_size = int(args.train_batch_size / args.gradient_accumulation_steps) // dist.get_world_size()
                train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler,
                                                            num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors, pin_memory=False)
                                                            
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()