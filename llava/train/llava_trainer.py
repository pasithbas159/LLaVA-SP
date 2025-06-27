import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in
                    get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i: i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i: i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                          generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size,
                                                 generator=self.generator)
        return iter(indices)


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                projector_parameters2 = [name for name, _ in opt_model.named_parameters() if "conv_linear" in name]
                projector_parameters3 = [name for name, _ in opt_model.named_parameters() if "tfm_linear" in name]
                projector_parameters4 = [name for name, _ in opt_model.named_parameters() if "CF_linear" in name]
                projector_parameters5 = [name for name, _ in opt_model.named_parameters() if "PF_linear" in name]
                projector_parameters6 = [name for name, _ in opt_model.named_parameters() if "conv_linear_f" in name]

                projector_parameters7 = [name for name, _ in opt_model.named_parameters() if "query_projector" in name]
                projector_parameters8 = [name for name, _ in opt_model.named_parameters() if "key_projector" in name]
                projector_parameters9 = [name for name, _ in opt_model.named_parameters() if "value_projector" in name]

                projector_parameters10 = [name for name, _ in opt_model.named_parameters() if
                                          "query_projector_small" in name]
                projector_parameters11 = [name for name, _ in opt_model.named_parameters() if
                                          "key_projector_small" in name]
                projector_parameters12 = [name for name, _ in opt_model.named_parameters() if
                                          "value_projector_small" in name]

                projector_parameters13 = [name for name, _ in opt_model.named_parameters() if
                                          "query_projector_mid" in name]
                projector_parameters14 = [name for name, _ in opt_model.named_parameters() if
                                          "key_projector_mid" in name]
                projector_parameters15 = [name for name, _ in opt_model.named_parameters() if
                                          "value_projector_mid" in name]

                projector_parameters16 = [name for name, _ in opt_model.named_parameters() if
                                          "query_projector_huge" in name]
                projector_parameters17 = [name for name, _ in opt_model.named_parameters() if
                                          "key_projector_huge" in name]
                projector_parameters18 = [name for name, _ in opt_model.named_parameters() if
                                          "value_projector_huge" in name]

                projector_parameters = (projector_parameters + projector_parameters2 + projector_parameters3
                                        + projector_parameters4 + projector_parameters5 + projector_parameters6
                                        + projector_parameters7 + projector_parameters8 + projector_parameters9
                                        + projector_parameters10 + projector_parameters11 + projector_parameters12
                                        + projector_parameters13 + projector_parameters14 + projector_parameters15
                                        + projector_parameters16 + projector_parameters17 + projector_parameters18)
                conv2_parameters = [name for name, _ in opt_model.named_parameters() if "conv_2" in name]
                conv4_parameters = [name for name, _ in opt_model.named_parameters() if "conv_4" in name]
                conv6_parameters = [name for name, _ in opt_model.named_parameters() if "conv_6" in name]
                conv8_parameters = [name for name, _ in opt_model.named_parameters() if "conv_8" in name]
                conv10_parameters = [name for name, _ in opt_model.named_parameters() if "conv_10" in name]
                conv12_parameters = [name for name, _ in opt_model.named_parameters() if "conv_12" in name]
                conv14_parameters = [name for name, _ in opt_model.named_parameters() if "conv_14" in name]
                conv16_parameters = [name for name, _ in opt_model.named_parameters() if "conv_16" in name]
                conv18_parameters = [name for name, _ in opt_model.named_parameters() if "conv_18" in name]
                conv20_parameters = [name for name, _ in opt_model.named_parameters() if "conv_20" in name]
                conv22_parameters = [name for name, _ in opt_model.named_parameters() if "conv_22" in name]
                conv24_parameters = [name for name, _ in opt_model.named_parameters() if "conv_24" in name]

                conv8_p_parameters = [name for name, _ in opt_model.named_parameters() if "conv_8_p" in name]
                conv16_p_parameters = [name for name, _ in opt_model.named_parameters() if "conv_16_p" in name]
                conv24_p_parameters = [name for name, _ in opt_model.named_parameters() if "conv_24_p" in name]

                convcm_parameters = [name for name, _ in opt_model.named_parameters() if "conv_CM" in name]

                convsmall_parameters = [name for name, _ in opt_model.named_parameters() if "conv_small" in name]
                convmid_parameters = [name for name, _ in opt_model.named_parameters() if "conv_mid" in name]
                convhuge_parameters = [name for name, _ in opt_model.named_parameters() if "conv_huge" in name]

                cmp_parameters = [name for name, _ in opt_model.named_parameters() if "CM_P" in name]
                cml_parameters = [name for name, _ in opt_model.named_parameters() if "CM_L" in name]
                conv_parameters = (conv8_p_parameters + conv16_p_parameters+ conv24_p_parameters + conv2_parameters + conv4_parameters + conv6_parameters+ conv8_parameters
                                   + conv10_parameters+ conv12_parameters + conv14_parameters+ conv16_parameters
                                   + conv18_parameters+ conv20_parameters + conv22_parameters+ conv24_parameters + cml_parameters + cmp_parameters + convcm_parameters
                                   + convsmall_parameters + convmid_parameters + convhuge_parameters)

                tfm4_parameters = [name for name, _ in opt_model.named_parameters() if "tfm_4" in name]
                tfm8_parameters = [name for name, _ in opt_model.named_parameters() if "tfm_8" in name]
                tfm12_parameters = [name for name, _ in opt_model.named_parameters() if "tfm_12" in name]
                tfm16_parameters = [name for name, _ in opt_model.named_parameters() if "tfm_16" in name]
                tfm20_parameters = [name for name, _ in opt_model.named_parameters() if "tfm_20" in name]
                tfm24_parameters = [name for name, _ in opt_model.named_parameters() if "tfm_24" in name]
                tfm_parameters = (tfm4_parameters + tfm8_parameters + tfm12_parameters + tfm16_parameters
                                  + tfm20_parameters + tfm24_parameters)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters
                                                                           and n not in projector_parameters
                                                                           and n not in conv_parameters

                                                                           and n not in tfm_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters
                                                                           and n not in projector_parameters
                                                                           and n not in conv_parameters

                                                                           and n not in tfm_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },

                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters
                                                                           and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },

                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters
                                                                           and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },

                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in conv_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in conv_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },

                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in tfm_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in tfm_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },

                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler', 'conv_CM', 'conv_4', 'CM_P', 'conv_linear']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
