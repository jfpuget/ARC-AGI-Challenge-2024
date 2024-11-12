from types import SimpleNamespace
from pathlib import Path

cfg = SimpleNamespace(**{})

cfg.gpu = "0"
cfg.seed = 0
cfg.fname = 'gen10000'

cfg.data_path = Path("../re-arc/%s/tasks" % cfg.fname )
cfg.input_path = Path("../input/arc-prize-2024/")
cfg.save_path = Path('../checkpoints')
cfg.checkpoint_path = ""
cfg.pretrained_path = '../checkpoints/ngc/exp_54.pt'

# training

# scale
size = 'large' # 'gpt2'
if size == 'small':
    cfg.sample_batch_size = 1
    cfg.train_batch_size = 64
    cfg.valid_batch_size = 128
    cfg.num_epochs = 1
    cfg.hidden_dim = 512
    cfg.heads = 8
    cfg.depth = 6
    cfg.lr =  1e-5
elif size == 'large':
    cfg.sample_batch_size = 1
    cfg.train_batch_size = 32
    cfg.valid_batch_size = 64
    cfg.num_epochs = 1
    cfg.hidden_dim = 512
    cfg.heads = 8
    cfg.depth = 12
    cfg.lr =  5e-6
elif size == 'gpt2':
    cfg.sample_batch_size = 1
    cfg.train_batch_size = 32
    cfg.valid_batch_size = 64
    cfg.num_epochs = 1
    cfg.hidden_dim = 768
    cfg.heads = 12
    cfg.depth = 12
    cfg.lr =  5e-6
    
# training
cfg.fp16 = True
cfg.accumulate = 1
cfg.grad_norm = False
cfg.grad_norm_type = 2
cfg.device = "cuda"
cfg.ddp = False
cfg.ddp_valid = False
cfg.local_rank = 0
cfg.workers = 1

# task

cfg.train_model = False
cfg.tune_model = True
cfg.predict_model = False
cfg.logger_file = True
cfg.verbose = 0
cfg.loss_length = 100
cfg.valid = 100
cfg.mean_loss = True

# model

cfg.num_color = 10
cfg.num_task = 900
cfg.max_size = 30
cfg.use_embed = False
cfg.tied_embedding = True
cfg.dim_heads = cfg.hidden_dim // cfg.heads
cfg.dropout = 0.
cfg.repeat_layers = False
cfg.norm_input = False
cfg.manual_norm_weights = False
cfg.tie_task_embed = False
cfg.task_embed_size = 32
cfg.qk_scale = True
cfg.aug_sym = False
cfg.aug_transpose = True
cfg.aug_color = 8
cfg.eval_aug_color = 8
cfg.eval_aug_sym = False
cfg.eval_aug_transpose = True
cfg.eval_task_embed_size = 32
cfg.freeze = False
cfg.num_train_task = 300
cfg.fuse_task = True
cfg.fold = -1

# optimizer

cfg.opt_beta1 = 0.9
cfg.opt_beta2 = 0.999
cfg.opt_eps = 1e-8
cfg.optimizer = 'AdamW'

# scheduler

cfg.scheduler = 'cosine'
cfg.pct_start = 0.01
cfg.final_div_factor = 1e-4

