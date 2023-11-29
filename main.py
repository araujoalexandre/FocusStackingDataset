
import os
import sys
import shutil
import argparse
import glob
import copy
from datetime import datetime
from os import makedirs
from os.path import join, exists, realpath, expandvars, basename

import submitit
from core.utils import Config


def main(config):

  # default: set tasks_per_node equal to number of gpus
  tasks_per_node = config.cluster.ngpus
  if config.cluster.single_node or config.project.mode == 'eval':
    tasks_per_node = 1
  cluster = 'slurm' if not config.cluster.local else 'local'

  executor = submitit.AutoExecutor(
    folder=config.project.train_dir, cluster=cluster)
  executor.update_parameters(
    gpus_per_node=config.cluster.ngpus,
    nodes=config.cluster.nnodes,
    tasks_per_node=tasks_per_node,
    cpus_per_task=40 // tasks_per_node,
    stderr_to_stdout=True,
    slurm_account=config.cluster.account,
    slurm_job_name=f'{config.project.train_dir[-4:]}_{config.project.mode}',
    slurm_partition=config.cluster.partition,
    slurm_qos=config.cluster.qos,
    slurm_constraint=config.cluster.constraint,
    slurm_signal_delay_s=0,
    timeout_min=config.cluster.timeout,
  )

  if config.cluster.begin:
    executor.update_parameters(
      slurm_additional_parameters={'begin': config.cluster.begin},
    )

  if config.project.mode == 'train':
    from core.trainer import Trainer
    trainer = Trainer(config)
    job = executor.submit(trainer)
  elif config.project.mode == 'eval':
    from core.evaluate import Evaluator
    if config.eval.burst_name == 'all':
      burst_path = glob.glob(join(config.eval.eval_data_dir, '*'))
      burst_names = [x.split('/')[-1] for x in burst_path]
      with executor.batch():
        for burst_name in burst_names:
          new_config = copy.deepcopy(config)
          new_config.eval.burst_name = burst_name
          evaluate = Evaluator(new_config)
          job = executor.submit(evaluate)
    else:
      evaluate = Evaluator(config)
      job = executor.submit(evaluate)

  folder = config.project.train_dir.split('/')[-1]
  print(f"Submitted batch job {job.job_id} in folder {folder}")



if __name__ == '__main__':

  parser = argparse.ArgumentParser(description="Super Resolution & Focus Stacking.")
  
  # cluster settings
  parser_cluster = parser.add_argument_group("cluster", "Cluster configurations")
  parser_cluster.add_argument("--account", type=str, help="Account to use for slurm.")
  parser_cluster.add_argument("--ngpus", type=int, default=4, help="Number of GPUs to use.")
  parser_cluster.add_argument("--nnodes", type=int, default=1, help="Number of nodes.")
  parser_cluster.add_argument("--timeout", type=int, default=1200, help="Time of the Slurm job in minutes for training.")
  parser_cluster.add_argument("--partition", type=str, help="Partition to use for Slurm.")
  parser_cluster.add_argument("--qos", type=str, help="Choose Quality of Service for slurm job.")
  parser_cluster.add_argument("--constraint", type=str, default=None, help="Add constraint for choice of GPUs: 16 or 32")
  parser_cluster.add_argument("--begin", type=str, default='', help="Set time to begin job")
  parser_cluster.add_argument("--local", action='store_true', help="Execute with local machine instead of slurm.")
  parser_cluster.add_argument("--debug", action="store_true", help="Activate debug mode.")
  # default training is distributed (even on single node)
  # to force non-distributed training, activate single node
  parser_cluster.add_argument("--single-node", action='store_true', help="Force non-distributed training.")
  
  parser_project = parser.add_argument_group("project", "Project parameters")
  parser_project.add_argument("--mode", type=str, default="train", choices=['train', 'eval'], 
                              help="Choice of the mode of execution.")
  parser_project.add_argument("--train_dir", type=str,
                              help="Name of the training directory.")
  parser_project.add_argument("--data_dir", type=str,
                              help="Name of the data directory.")
  parser_project.add_argument("--dataset", type=str, default='focus_stack_dataset')
  parser_project.add_argument("--dataset_id", type=str, default='_raw') # 3  = JPG / '_raw' = RAW
  parser_project.add_argument("--seed", type=int,
                              help="Make the training deterministic.")
  parser_project.add_argument("--logging_verbosity", type=str, default='INFO',
                              help="Choose the level of verbosity of the logs")
  parser_project.add_argument("--autocast", type=eval, default=True,
                              help="Enable mix precision training.")
  
  # parameters training
  parser_training = parser.add_argument_group("training", "Training parameters")
  parser_training.add_argument("--epochs", type=int, default=300,
                               help="Number of epochs for training.")
  parser_training.add_argument("--optimizer", type=str, default="adam",
                               help="Optimizer to use for training.")
  parser_training.add_argument("--wd", type=float, default=0,
                               help="Weight decay to use for training.")
  parser_training.add_argument("--lr", type=float, default=0.0001,
                               help="The learning rate to use for training.")
  parser_training.add_argument("--decay", type=str, default='100-200',
                               help="Decay to use for the learning rate schedule.")
  parser_training.add_argument("--gamma", type=float, default=0.5,
                               help="Gamma value to use for learning rate schedule.")
  parser_training.add_argument("--gradient_clip_by_norm", type=float, default=None,
                               help="Clip the norm of the gradient during training.")
  parser_training.add_argument("--gradient_clip_by_value", type=float, default=None,
                               help="Clip the gradient by value during training.")
  parser_training.add_argument("--batch_size", type=int, default=8,
                               help="The batch size to use for training.")
  parser_training.add_argument("--frequency_log_steps", type=int, default=10,
                               help="Print log for every step.")
  parser_training.add_argument("--save_checkpoint_epochs", type=int, default=1,
                               help="Save checkpoint every epoch.")
  
  # parameters eval
  parser_eval = parser.add_argument_group("eval", "Evaluation parameters")
  parser_eval.add_argument("--dataset_split", type=str)
  parser_eval.add_argument("--dataset_lens", type=str, default='lumix_lens',
                           choices=['lumix_lens', 'olympus_macro_lens', ''])
  parser_eval.add_argument("--prediction_folder", type=str, default="./imsave")
  parser_eval.add_argument("--image_type", type=str, default='raw', choices=['raw', 'jpg', 'aligned', ''])
  parser_eval.add_argument("--image_scaling", type=float, default=1)
  parser_eval.add_argument("--burst_name", type=str, default="all")
  parser_eval.add_argument("--eval_batch_size", type=int, default=8, help="The batch size to use for eval.")
  parser_eval.add_argument("--patch_size", type=int, default=256, help="size of the tile")
  parser_eval.add_argument('--align', type=eval, default=1)
  parser_eval.add_argument('--eval_by_patch', type=eval, default=1)
  # optical flow parameters
  parser_eval.add_argument("--of_scale", type=int, default=4)
  parser_eval.add_argument("--of_win", type=int, default=300)

  parser_eval.add_argument("--noise", type=int, default=None)

  # parameters for data generation pipeline
  parser_data = parser.add_argument_group("data", "Data pipeline parameters.")
  parser_data.add_argument("--crop_sz", type=int, default=64,
                           help="Size of crop to use for data generation.")
  parser_data.add_argument("--burst_size", type=int, default=30,
                           help="Size of the burst to use for data generation.")
  parser_data.add_argument("--max_translation", type=float, default=0.,
                           help='Max translation to use during Synthetic Burst Generation.')
  parser_data.add_argument('--max_rotation', type=float, default=0.,
                           help='Max rotation to use during Synthetic Burst Generation.')
  parser_data.add_argument('--max_scale', type=float, default=0.,
                           help='Max scale to use during Synthetic Burst Generation.')
  parser_data.add_argument('--border_crop', type=int, default=24,
                           help='Border crop to use during Synthetic Burst Generation.')
  parser_data.add_argument("--downsample_factor", type=int, default=1,
                           help="For factor > 1, train for superresolution")
  parser_data.add_argument("--overfit", type=str, default=None)
  parser_data.add_argument("--convert-to-raw", action='store_true', default=False)
  parser_data.add_argument("--add_noise", action='store_true')
  
  # parameters of the model architecture
  parser_archi = parser.add_argument_group("archi", "Model architecture parameters.")
  parser_archi.add_argument("--model", default='ebsr', type=str)
  parser_archi.add_argument("--n_channels", type=int, default=64)
  parser_archi.add_argument("--fusion", type=str, default="conv",
                            help="Define the fusion block")
  parser_archi.add_argument("--n_feats", type=int, default=64,
                            help='number of feature maps')
  parser_archi.add_argument("--n_resblocks", type=int, default=4,
                            help='number of residual blocks')
  parser_archi.add_argument("--n_resgroups", type=int, default=5,
                            help='number of residual groups')
  parser_archi.add_argument("--lrcn", type=eval, default=True,
                            help='use long-range concatenating network')
  parser_archi.add_argument("--n_colors", type=int, default=1,
                            help='number of color channels to use')
  
  # parse all arguments and define config object
  args = parser.parse_args()
  arg_groups={}
  for group in parser._action_groups:
    group_dict={a.dest:getattr(args,a.dest,None) for a in group._group_actions}
    arg_groups[group.title] = argparse.Namespace(**group_dict)
 
  # create config object
  config = Config(**arg_groups)
  config.cmd = f"python3 {' '.join(sys.argv)}"

  if config.project.mode == 'eval':
    if config.project.dataset == 'focus_stack_dataset':
      data_dir = "./data/focus_stack_dataset/dataset/"
      config.eval.eval_data_dir = join(data_dir, config.eval.dataset_split, config.eval.dataset_lens)
    else:
      config.eval.eval_data_dir = f'./data/{config.project.dataset}'

  # get the path of datsets
  if config.project.data_dir is None:
    config.project.data_dir = os.environ.get('DATADIR', None)
  if config.project.data_dir is None:
    ValueError("the following arguments are required: --data_dir")
  
  # set the partition for debug jobs
  if config.cluster.debug:
    config.cluster.qos = "qos_gpu-dev"
    config.cluster.timeout = 60
  
  # path to folder where to saved models
  path = realpath('./trained_models')
  # create models folder if it does not exist
  makedirs(path, exist_ok=True)

  if config.project.train_dir is not None:
    config.project.train_dir = f'{path}/{config.project.train_dir}'
    if not exists(config.project.train_dir):
      ValueError('The train directory cannot be found.')
 
  # create the folder for the training
  if config.project.mode == 'train' and config.project.train_dir is None:
    config.training.start_new_model = True
    folder = datetime.now().strftime("%Y-%m-%d_%H.%M.%S_%f")[:-2]
    if config.cluster.debug:
      folder = 'folder_debug'
      # delete this folder 
      shutil.rmtree(f'{path}/{folder}', ignore_errors=True)
    config.project.train_dir = f'{path}/{folder}'
    makedirs(config.project.train_dir)
    makedirs(f'{config.project.train_dir}/checkpoints')
  elif config.project.mode == 'train' and config.project.train_dir is not None:
    config.training.start_new_model = False
    assert exists(config.project.train_dir)

  if config.project.mode == "eval":
    if config.project.train_dir is None:
      ValueError('The train directory must be provided.')
    train_folder = basename(config.project.train_dir)
    if config.project.dataset == 'focus_stack_dataset':
      config.eval.prediction_folder = join(
        realpath(config.eval.prediction_folder), train_folder, 
        'focus_stack_dataset', config.eval.dataset_split, config.eval.dataset_lens)
    else:
      config.eval.prediction_folder = join(
        realpath(config.eval.prediction_folder), train_folder, config.project.dataset)
    makedirs(config.eval.prediction_folder, exist_ok=True)

  main(config)



