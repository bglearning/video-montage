import argparse
import time

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from transformers.optimization import AdamW
import wandb  # Quit early if user doesn't have wandb installed.

from models.pretrain_dataset import AllDataset
from models import CoTransformer, CoTransformerED
from tasks import SequenceMatching, SequenceRecovery


def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]


def setup_scheduler(opt, epochs, transition_epochs=10):
    warmup_scheduler = LinearLR(opt, total_iters=transition_epochs)

    cosine_lr = CosineAnnealingWarmRestarts(
        opt,
        T_0=epochs,
        T_mult=1,
        eta_min=1e-6
    ) # Minimum learning rate

    scheduler = SequentialLR(
        opt,
        schedulers=[warmup_scheduler, cosine_lr],
        milestones=[transition_epochs]
    )

    return scheduler


def setup_optimizer(transformer, learning_rate):
    lr = learning_rate
    wd = 0.01

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = 1

    optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in transformer.named_parameters()
                    if any(nd in n for nd in no_decay)
                    and not any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr,
            },
            {
                "params": [
                    p
                    for n, p in transformer.named_parameters()
                    if not any(nd in n for nd in no_decay)
                    and any(bb in n for bb in head_names)
                ],
                "weight_decay": wd,
                "lr": lr * lr_mult,
            },
            {
                "params": [
                    p
                    for n, p in transformer.named_parameters()
                    if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
                ],
                "weight_decay": 0.0,
                "lr": lr * lr_mult,
            },
        ]

    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8, betas=(0.9, 0.98))


def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=False)

    group.add_argument('--model_load_path', default='', type=str,
                    help='path to your trained Transformer')

    group.add_argument('--transformer_path', type=str,
                    help='path to your partially trained Transformer')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='path to your folder of frames')
    parser.add_argument('--frame_suffix', type=str, default = '')
    parser.add_argument('--embed_dim', type=int, default =512)

    parser.add_argument('--json_file', type=str, required=True,
                        help='path to your json file of captions and shots')
    parser.add_argument('--caption_file', type=str, required=False,
                        help='path to the video captions')
    parser.add_argument('--dataset_type', type=str, required=False, default='vspd', choices=('vspd', 'tiktok', 'tiktok_caps'),
                        help='Type of dataset')

    parser.add_argument('--transformer_output_file_name', type=str, default = "checkpoints/transformer_pretrained",
                        help='output_file_name')
                        
    parser.add_argument('--seed', type=int, default=1024, help='Seed for random number')

    parser.add_argument('--wandb_name', default='dalle_train_transformer',
                        help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')

    parser.add_argument('--wandb_entity', default=None,
                        help='(optional) Name of W&B team/entity to log to.')

    train_group = parser.add_argument_group('Training settings')

    train_group.add_argument('--phase', default = 'train', type = str, help = 'train or test')

    train_group.add_argument("--seq_len", type=int, default=10, help="Max length of sequence")

    train_group.add_argument('--epochs', default = 50, type = int, help = 'Number of epochs')

    train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')

    train_group.add_argument('--keep_n_checkpoints', default = None, type = int, help = '(Careful) Deletes old deepspeed checkpoints if there are more than n')

    train_group.add_argument('--batch_size', default = 64, type = int, help = 'Batch size')

    train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')

    train_group.add_argument('--learning_rate', default = 5e-5, type = float, help = 'Learning rate')

    train_group.add_argument('--loss_weight', default = 1.0, type = float, help = 'weight to balance the loss')

    train_group.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')

    train_group.add_argument('--lr_decay', dest = 'lr_decay', action = 'store_true')

    model_group = parser.add_argument_group('Model settings')

    model_group.add_argument('--clip_model', default = "ViT-B/32", type = str, help = 'Name of CLIP')

    model_group.add_argument('--neg_sampler', default = "full_random", type = str, help = 'Negative sampler')
    model_group.add_argument('--similarities_file', default =None, type = str, help = 'File with similarity matrix')

    model_group.add_argument('--hidden_size', default = 512, type = int, help = 'Model dimension')

    model_group.add_argument('--image_size', default = 256, type = int, help = 'Size of image')

    model_group.add_argument('--mlp_ratio', default = 4, type = int, help = 'mlp_ratio')

    model_group.add_argument('--drop_rate', default = 0.1, type = float, help = 'drop_rate')

    model_group.add_argument('--num_heads', default = 8, type = int, help = 'Model number of heads')

    model_group.add_argument('--num_layers', default = 2, type = int, help = 'Model depth')

    model_group.add_argument('--topk', default = 4, type = int)

    model_group.add_argument("--nce_T", type=float, default=0.05, help="Temperature for nec Loss")

    model_group.add_argument("--ratio", type=float, default=0.15, help="Ratio for random mask")

    model_group.add_argument("--split-train", type=bool, default=True, help="Whether to split train-val")

    args = parser.parse_args()

    # constants
    TRANSFORMER_OUTPUT_FILE_NAME = args.transformer_output_file_name + ".pt"
    Path(TRANSFORMER_OUTPUT_FILE_NAME).parent.mkdir(exist_ok=True)

    TRANSFORMER_PATH = args.transformer_path
    RESUME = exists(TRANSFORMER_PATH)

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    LEARNING_RATE = args.learning_rate
    assert Path(args.data_dir).exists(), f'The path {args.data_dir} was not found.'

    # reconstitute vae
    if RESUME:
        transformer_path = Path(TRANSFORMER_PATH)
        
        assert transformer_path.exists(), 'TRANSFORMER model file does not exist'
        
        loaded_obj = torch.load(str(transformer_path), map_location='cpu')

        transformer_params, weights = loaded_obj['hparams'], loaded_obj['weights']
        opt_state = loaded_obj.get('opt_state')
        scheduler_state = loaded_obj.get('scheduler_state')

        transformer_params = dict(
            **transformer_params
        )
        resume_epoch = loaded_obj.get('epoch', 0)
        print("load partially model")
    else:
        transformer_params = dict(
            args = args
        )
        resume_epoch = 0

    # create dataset and dataloader

    ds = AllDataset(
        args=args
    )
    assert len(ds) > 0, 'dataset is empty'

    val_args = argparse.Namespace(**vars(args))
    # All args same except source json_file, caption_file
    # And neg_sampler
    val_args.json_file = val_args.json_file.replace("train", "val")
    if val_args.caption_file is not None:
        val_args.caption_file = val_args.caption_file.replace("train", "val")
    val_args.neg_sampler = "full_random"
    val_ds = AllDataset(args=val_args)

    print(f'{len(ds)} image-text pairs set for training')

    # Regular DataLoader for image-text-folder datasets
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, sampler=None)

    if args.split_train:
        val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, sampler=None)

    transformer = CoTransformer(**transformer_params)
    transformer = transformer.cuda()

    if RESUME:
        transformer.load_state_dict(weights)

    # optimizer
    opt = setup_optimizer(transformer=transformer, learning_rate=LEARNING_RATE)
    scheduler = setup_scheduler(opt, epochs=EPOCHS, transition_epochs=10)

    run = wandb.init(
        project=args.wandb_name,
        entity=args.wandb_entity,
        resume=False,
        config=vars(args),
    )

    print("load all model")

    def save_model(path, epoch=0):
        save_obj = {
            'hparams': transformer_params,
            'epoch': epoch,
        }

        save_obj = {
            **save_obj,
            'weights': transformer.state_dict(),
            'opt_state': opt.state_dict(),
        }
        save_obj['scheduler_state'] = (scheduler.state_dict() if scheduler else None)
        torch.save(save_obj, path)

    # training

    save_model(TRANSFORMER_OUTPUT_FILE_NAME, epoch=resume_epoch)

    sequence_matching = SequenceMatching(seq_len=args.seq_len)
    sequence_recovery = SequenceRecovery()

    def run_one_epoch(epoch_dl, sequence_matching_task, sequence_recovery_task, n_epoch: int, train: bool=True):

        pre_tag = 'val_' if not train else ''

        for i, (shots, false_shots, texts, shot_mask, false_mask, text_mask) in enumerate(epoch_dl):
            if i % 10 == 0:
                t = time.time()
            
            itm_shots, itm_mask, itm_labels = sequence_matching_task.create(shots, shot_mask, false_shots, false_mask)
            shuffle_shots, s_mask, s_labels = sequence_recovery_task.create(shots, shot_mask)
            
            texts = texts.cuda()
            text_mask = text_mask.cuda()

            if not train:
                transformer.eval()
            
            torch.set_grad_enabled(train)
            
            itm_loss, shuffle_loss = transformer(itm_shots, shuffle_shots, texts, itm_labels, s_labels, text_mask, itm_mask, s_mask)
            
            loss = itm_loss + args.loss_weight * shuffle_loss

            if train:
                torch.set_grad_enabled(True)
                loss.backward()
                opt.step()
                opt.zero_grad()

            log = {}

            if i % 10 == 0:
                print(epoch, i, f'loss - {loss.item()}')
                
                log = {
                    **log,
                    'epoch': epoch,
                    f'{pre_tag}iter': i,
                    f'{pre_tag}loss': loss.item(),
                    f'{pre_tag}itm_loss': itm_loss.item(),
                    f'{pre_tag}shuffle_loss': shuffle_loss.item(),
                }

            if train:
                log['lr'] = opt.state_dict()['param_groups'][0]['lr']
                if i % 10 == 9:
                    sample_per_sec = BATCH_SIZE * 10 / (time.time() - t)
                    log["sample_per_sec"] = sample_per_sec
                    print(epoch, i, f'sample_per_sec - {sample_per_sec}')

            wandb.log(log) 
            
        if train:
            scheduler.step()

    for epoch in range(resume_epoch, EPOCHS):
        if args.split_train:
            run_one_epoch(epoch_dl=val_dl,
                          sequence_matching_task=sequence_matching,
                          sequence_recovery_task=sequence_recovery,
                          n_epoch=epoch, train=False)
        run_one_epoch(epoch_dl=dl,
                      sequence_matching_task=sequence_matching,
                      sequence_recovery_task=sequence_recovery,
                      n_epoch=epoch, train=True)
        
        epoch_name = TRANSFORMER_OUTPUT_FILE_NAME.split('.')[0] + '_' + str(epoch) + '.pt'
        if epoch % 10 == 0:
            save_model(epoch_name, epoch=epoch)
        
    save_model(TRANSFORMER_OUTPUT_FILE_NAME, epoch=epoch)
    wandb.finish()


if __name__ == '__main__':
    main()