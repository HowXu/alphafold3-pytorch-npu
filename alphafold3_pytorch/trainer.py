from __future__ import annotations

from functools import wraps, partial
from contextlib import contextmanager
from pathlib import Path

from alphafold3_pytorch.alphafold3 import Alphafold3

from beartype.typing import TypedDict, List, Callable

from alphafold3_pytorch.tensor_typing import (
    should_typecheck,
    typecheck,
    Int, Bool, Float
)

from alphafold3_pytorch.inputs import (
    AtomInput,
    BatchedAtomInput,
    Alphafold3Input,
    PDBInput,
    maybe_transform_to_atom_inputs,
    alphafold3_inputs_to_batched_atom_input,
    collate_inputs_to_batched_atom_input,
    UNCOLLATABLE_ATOM_INPUT_FIELDS,
    ATOM_DEFAULT_PAD_VALUES,
)

from alphafold3_pytorch.data import (
    mmcif_writing
)

import torch
from torch import tensor
from torch.nn import Module
from torch.optim import Adam, Optimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler, Dataset, DataLoader as OrigDataLoader
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

from lion_pytorch.foreach import Lion
from adam_atan2_pytorch.foreach import AdamAtan2
from adam_atan2_pytorch.adopt_atan2 import AdoptAtan2

from ema_pytorch import EMA

from lightning import Fabric
from lightning.fabric.loggers import Logger
from lightning.fabric.wrappers import _unwrap_objects

from shortuuid import uuid

# helpers

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

def at_most_one_of(*flags: bool) -> bool:
    return sum([*map(int, flags)]) <= 1

@contextmanager
def to_device_and_back(
    module: Module,
    device: torch.device
):
    orig_device = next(module.parameters()).device
    need_move_device = orig_device != device

    if need_move_device:
        module.to(device)

    yield

    if need_move_device:
        module.to(orig_device)

def cycle(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch

@typecheck
def accum_dict(
    past_losses: dict | None,
    losses: dict,
    scale: float = 1.
):
    losses = {k: v * scale for k, v in losses.items()}

    if not exists(past_losses):
        return losses

    for loss_name in past_losses.keys():
        past_losses[loss_name] += losses.get(loss_name, 0.)

    return past_losses

# dataloader and collation fn

@typecheck
def DataLoader(
    *args,
    atoms_per_window: int | None = None,
    map_input_fn: Callable | None = None,
    transform_to_atom_inputs: bool = True,
    **kwargs
):
    collate_fn = partial(
        collate_inputs_to_batched_atom_input,
        atoms_per_window = atoms_per_window,
        transform_to_atom_inputs = transform_to_atom_inputs,
    )

    if exists(map_input_fn):
        collate_fn = partial(collate_fn, map_input_fn = map_input_fn)

    return OrigDataLoader(*args, collate_fn = collate_fn, **kwargs)

# default scheduler used in paper w/ warmup

def default_lambda_lr_fn(steps):
    # 1000 step warmup

    if steps < 1000:
        return steps / 1000

    # decay 0.95 every 5e4 steps

    steps -= 1000
    return 0.95 ** (steps / 5e4)

# main class

class Trainer:
    """ Section 5.4 """

    @typecheck
    def __init__(
        self,
        model: Alphafold3,
        *,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int,
        grad_accum_every: int = 1,
        map_dataset_input_fn: Callable | None = None,
        valid_dataset: Dataset | None = None,
        valid_every: int = 1000,
        test_dataset: Dataset | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        ema_decay = 0.999,
        lr = 1.8e-3,
        default_adam_kwargs: dict = dict(
            betas = (0.9, 0.95),
            eps = 1e-8
        ),
        clip_grad_norm = 10.,
        default_lambda_lr = default_lambda_lr_fn,
        train_sampler: Sampler | None = None,
        fabric: Fabric | None = None,
        loggers: List[Logger] = [],
        accelerator = 'auto',
        checkpoint_prefix = 'af3.ckpt.',
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        overwrite_checkpoints: bool = False,
        fabric_kwargs: dict = dict(),
        distributed_eval: bool = True,
        fp16: bool = False,
        use_ema: bool = True,
        ema_kwargs: dict = dict(
            use_foreach = True
        ),
        ema_on_cpu = False,
        ema_update_model_with_ema_every: int | None = None,
        use_adam_atan2: bool = False,
        use_adopt_atan2: bool = False,
        use_lion: bool = False,
        use_torch_compile: bool = False
    ):
        super().__init__()

        # fp16 precision is a root level kwarg

        if fp16:
            assert 'precision' not in fabric_kwargs
            fabric_kwargs.update(precision = '16-mixed')

        # instantiate fabric

        if not exists(fabric):
            fabric = Fabric(
                accelerator = accelerator,
                loggers = loggers,
                **fabric_kwargs
            )

        self.fabric = fabric
        fabric.launch()

        # whether evaluating only on root node or not
        # to save on each machine keeping track of EMA

        self.distributed_eval = distributed_eval
        self.will_eval_or_test = self.is_main or distributed_eval

        # using "switch" ema

        self.switch_ema = exists(ema_update_model_with_ema_every)

        # exponential moving average

        self.ema_model = None
        self.has_ema = (self.will_eval_or_test or self.switch_ema) and use_ema

        if self.has_ema:
            self.ema_model = EMA(
                model,
                beta = ema_decay,
                include_online_model = False,
                allow_different_devices = True,
                coerce_dtype = True,
                update_model_with_ema_every = ema_update_model_with_ema_every,
                **ema_kwargs
            )

            self.ema_device = 'cpu' if ema_on_cpu else self.device
            self.ema_model.to(self.ema_device)

        # maybe torch compile

        if use_torch_compile:
            assert not should_typecheck, 'does not work well with jaxtyping + beartype, please invoke your training script with the environment flag `TYPECHECK=False` - ex. `TYPECHECK=False python train_af3.py`'
            model = torch.compile(model)

        # model

        self.model = model

        # optimizer

        if not exists(optimizer):
            optimizer_klass = Adam

            assert at_most_one_of(use_adam_atan2, use_adopt_atan2, use_lion)

            if use_adam_atan2:
                default_adam_kwargs.pop('eps', None)
                optimizer_klass = AdamAtan2
            elif use_adopt_atan2:
                default_adam_kwargs.pop('eps', None)
                optimizer_klass = AdoptAtan2
            elif use_lion:
                default_adam_kwargs.pop('eps', None)
                optimizer_klass = Lion

            optimizer = optimizer_klass(
                model.parameters(),
                lr = lr,
                **default_adam_kwargs
            )

        self.optimizer = optimizer

        # if map dataset function given, curry into DataLoader

        DataLoader_ = partial(DataLoader, atoms_per_window = model.atoms_per_window)

        if exists(map_dataset_input_fn):
            DataLoader_ = partial(DataLoader_, map_input_fn = map_dataset_input_fn)

        # maybe weighted sampling

        train_dl_kwargs = dict()

        if exists(train_sampler):
            train_dl_kwargs.update(sampler = train_sampler)
        else:
            train_dl_kwargs.update(
                shuffle = True,
                drop_last = True
            )

        # train dataloader

        self.dataloader = DataLoader_(
            dataset,
            batch_size = batch_size,
            **train_dl_kwargs
        )

        # validation dataloader on the EMA model

        self.valid_every = valid_every

        self.needs_valid = exists(valid_dataset)
        self.valid_dataloader = None

        if self.needs_valid and self.will_eval_or_test:
            self.valid_dataset_size = len(valid_dataset)
            self.valid_dataloader = DataLoader_(valid_dataset, batch_size = batch_size)

        # testing dataloader on EMA model

        self.needs_test = exists(test_dataset)
        self.test_dataloader = None

        if self.needs_test and self.will_eval_or_test:
            self.test_dataset_size = len(test_dataset)
            self.test_dataloader = DataLoader_(test_dataset, batch_size = batch_size)

        # training steps and num gradient accum steps

        self.num_train_steps = num_train_steps
        self.grad_accum_every = grad_accum_every

        # setup fabric

        self.model, self.optimizer = fabric.setup(self.model, self.optimizer)

        fabric.setup_dataloaders(self.dataloader)

        if exists(self.valid_dataloader) and self.distributed_eval:
            fabric.setup_dataloaders(self.valid_dataloader)

        if exists(self.test_dataloader) and self.distributed_eval:
            fabric.setup_dataloaders(self.test_dataloader)

        # scheduler

        if not exists(scheduler):
            scheduler = LambdaLR(optimizer, lr_lambda = default_lambda_lr)

        self.scheduler = scheduler

        # gradient clipping norm

        self.clip_grad_norm = clip_grad_norm

        # steps

        self.steps = 0

        # checkpointing logic

        self.checkpoint_prefix = checkpoint_prefix
        self.checkpoint_every = checkpoint_every
        self.overwrite_checkpoints = overwrite_checkpoints
        self.checkpoint_folder = Path(checkpoint_folder)

        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # save the path for the last loaded model, if any

        self.train_id = None

        self.last_loaded_train_id = None
        self.model_loaded_from_path: Path | None = None

    @property
    def device(self):
        return self.fabric.device

    @property
    def is_main(self):
        return self.fabric.global_rank == 0

    def generate_train_id(self):
        if exists(self.train_id):
            return

        self.train_id = uuid()[:4].lower()

    @property
    def train_id_with_prev(self) -> str:
        if not exists(self.last_loaded_train_id):
            return self.train_id

        ckpt_num = str(self.model_loaded_from_path).split('.')[-2]

        return f'{self.last_loaded_train_id}.{ckpt_num}-{self.train_id}'

    # saving and loading

    def save_checkpoint(self):
        assert exists(self.train_id_with_prev)

        # formulate checkpoint path and save

        checkpoint_path = self.checkpoint_folder / f'({self.train_id_with_prev})_{self.checkpoint_prefix}{self.steps}.pt'

        self.save(checkpoint_path, overwrite = self.overwrite_checkpoints)

    def save(
        self,
        path: str | Path,
        overwrite = False,
        prefix: str | None = None
    ):
        if isinstance(path, str):
            path = Path(path)

        assert not path.is_dir() and (not path.exists() or overwrite)

        path.parent.mkdir(exist_ok = True, parents = True)

        unwrapped_model = _unwrap_objects(self.model)
        unwrapped_optimizer = _unwrap_objects(self.optimizer)

        package = dict(
            model = unwrapped_model.state_dict_with_init_args,
            optimizer = unwrapped_optimizer.state_dict(),
            scheduler = self.scheduler.state_dict(),
            steps = self.steps,
            id = self.train_id
        )

        torch.save(package, str(path))

    def load_from_checkpoint_folder(
        self,
        **kwargs
    ):
        self.load(path = self.checkpoint_folder, **kwargs)

    def load(
        self,
        path: str | Path,
        strict = True,
        prefix = None,
        only_model = False,
        reset_steps = False
    ):
        if isinstance(path, str):
            path = Path(path)

        assert path.exists(), f'{str(path)} cannot be found for loading'

        # if the path is a directory, then automatically load latest checkpoint

        if path.is_dir():
            prefix = default(prefix, self.checkpoint_prefix)

            model_paths = [*path.glob(f'**/*_{prefix}*.pt')]

            assert len(model_paths) > 0, f'no files found in directory {path}'

            model_paths = sorted(model_paths, key = lambda p: int(str(p).split('.')[-2]))

            path = model_paths[-1]

        # get unwrapped model and optimizer

        unwrapped_model = _unwrap_objects(self.model)

        # load model from path

        model_id = unwrapped_model.load(path)

        # for eventually saving entire training history in filename

        self.model_loaded_from_path = path
        self.last_loaded_train_id = model_id

        if only_model:
            return

        # load optimizer and scheduler states

        package = torch.load(str(path), weights_only = True)

        unwrapped_optimizer = _unwrap_objects(self.optimizer)

        if 'optimizer' in package:
            unwrapped_optimizer.load_state_dict(package['optimizer'])

        if 'scheduler' in package:
            self.scheduler.load_state_dict(package['scheduler'])

        if reset_steps:
            self.steps = 0
        else:
            self.steps = package.get('steps', 0)

    # shortcut methods

    def wait(self):
        self.fabric.barrier()

    def print(self, *args, **kwargs):
        self.fabric.print(*args, **kwargs)

    def log(self, **log_data):
        self.fabric.log_dict(log_data, step = self.steps)

    # main train forwards

    def __call__(
        self
    ):

        self.generate_train_id()

        # cycle through dataloader

        dl = cycle(self.dataloader)

        # while less than required number of training steps

        while self.steps < self.num_train_steps:

            self.model.train()

            # gradient accumulation

            total_loss = 0.
            train_loss_breakdown = None

            for grad_accum_step in range(self.grad_accum_every):
                is_accumulating = grad_accum_step < (self.grad_accum_every - 1)

                inputs = next(dl)

                with self.fabric.no_backward_sync(self.model, enabled = is_accumulating):

                    # model forwards

                    loss, loss_breakdown = self.model(
                        **inputs.model_forward_dict(),
                        return_loss_breakdown = True
                    )

                    # accumulate

                    scale = self.grad_accum_every ** -1

                    total_loss += loss.item() * scale
                    train_loss_breakdown = accum_dict(train_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                    # backwards

                    self.fabric.backward(loss / self.grad_accum_every)

            # log entire loss breakdown

            self.log(**train_loss_breakdown)

            self.print(f'loss: {total_loss:.3f}')

            # clip gradients

            self.fabric.clip_gradients(self.model, self.optimizer, max_norm = self.clip_grad_norm)

            # optimizer step

            self.optimizer.step()

            # update exponential moving average

            self.wait()

            if self.has_ema:
                self.ema_model.update()

            self.wait()

            # scheduler

            self.scheduler.step()
            self.optimizer.zero_grad()

            self.steps += 1

            # maybe validate, for now, only on main with EMA model

            if (
                self.will_eval_or_test and
                self.needs_valid and
                divisible_by(self.steps, self.valid_every)
            ):
                eval_model = default(self.ema_model, self.model)

                with torch.no_grad(), to_device_and_back(eval_model, self.device):
                    eval_model.eval()

                    total_valid_loss = 0.
                    valid_loss_breakdown = None

                    for valid_batch in self.valid_dataloader:
                        valid_loss, loss_breakdown = eval_model(
                            **valid_batch.model_forward_dict(),
                            return_loss_breakdown = True
                        )

                        valid_batch_size = valid_batch.atom_inputs.shape[0]
                        scale = valid_batch_size / self.valid_dataset_size

                        total_valid_loss += valid_loss.item() * scale
                        valid_loss_breakdown = accum_dict(valid_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                    self.print(f'valid loss: {total_valid_loss:.3f}')

                # prepend valid_ to all losses for logging

                valid_loss_breakdown = {f'valid_{k}':v for k, v in valid_loss_breakdown.items()}

                # reduce valid loss breakdown

                if self.distributed_eval:
                    valid_loss_breakdown = self.fabric.all_reduce(valid_loss_breakdown, reduce_op = 'sum')

                # log

                self.log(**valid_loss_breakdown)

            self.wait()

            if self.is_main and divisible_by(self.steps, self.checkpoint_every):
                self.save_checkpoint()

            self.wait()

        # maybe test

        if self.will_eval_or_test and self.needs_test:
            eval_model = default(self.ema_model, self.model)

            with torch.no_grad(), to_device_and_back(eval_model, self.device):
                eval_model.eval()

                total_test_loss = 0.
                test_loss_breakdown = None

                for test_batch in self.test_dataloader:
                    test_loss, loss_breakdown = eval_model(
                        **test_batch.model_forward_dict(),
                        return_loss_breakdown = True
                    )

                    test_batch_size = test_batch.atom_inputs.shape[0]
                    scale = test_batch_size / self.test_dataset_size

                    total_test_loss += test_loss.item() * scale
                    test_loss_breakdown = accum_dict(test_loss_breakdown, loss_breakdown._asdict(), scale = scale)

                self.print(f'test loss: {total_test_loss:.3f}')

            # prepend test_ to all losses for logging

            test_loss_breakdown = {f'test_{k}':v for k, v in test_loss_breakdown.items()}

            # reduce

            if self.distributed_eval:
                test_loss_breakdown = self.fabric.all_reduce(test_loss_breakdown, reduce_op = 'sum')

            # log

            self.log(**test_loss_breakdown)

        print('training complete')
