import os
from re import L
import time
import logging
import pickle
import numpy as np
import sys
import random

from torch import nn
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import gc

from clip.clip import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import time

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": "ProDA",
                      "vision_depth": cfg.TRAINER.ProDA.N_PROMPT,
                      "vision_ctx": cfg.TRAINER.ProDA.N_CTX,
                      "vision_depth": 0,
                      "language_depth": 0,
                      "language_ctx": 0}
    assert cfg.TRAINER.ProDA.N_PROMPT >= 1, "For Vision Prompting, PROMPT_DEPTH_VISION should be >= 1"
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model.float()


class PromptLearner(nn.Module):
    def __init__(self, cfg, class_names, clip_model, n_ctx=16, n_prompt=32, prompt_bsz=4):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        # 将clip_model放到gpu上
        clip_model = clip_model

        n_cls = len(class_names)
        self.dtype = dtype
        ctx_vectors = torch.empty(n_prompt, n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        assert n_prompt % prompt_bsz == 0
        self.n_iter = int(n_prompt/prompt_bsz)

        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]

        classnames = [name.replace('_', ' ') for name in class_names]
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]

        if n_prompt >1:
            self.pos = [0 for _ in range(n_prompt//4)] + [1 for _ in range(n_prompt//4)] + [2 for _ in range(n_prompt//2)]
        else:
            self.pos = [2 for _ in range(n_prompt)]
        self.pos = torch.tensor(self.pos, device='cuda')

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype).cuda()
        self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer('token_suffix', embedding[:, 1+n_ctx:, :]) # CLS, EOS, [n_cls, -1, ctx_dim]

        nc_prompts = [prompt_prefix + '.' ]
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts).type(self.dtype).cuda()
        self.register_buffer('nc_token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
        self.register_buffer('nc_token_suffix', embedding[:, 1+n_ctx:, :]) # EOS, [n_cls, -1, ctx_dim]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.n_prompt = n_prompt
        self.ctx_dim = ctx_dim
        self.prompt_bsz = prompt_bsz
        self.prompt_build_mode = 'end'
        self.iter_idx = 0

    def forward(self, infer=False):

        if self.n_iter > 1 and (not infer):
            if self.iter_idx == 0:
                self.select_idx = torch.randperm(self.n_prompt, device='cuda')
            batch_idx = self.select_idx[self.iter_idx*self.prompt_bsz: (self.iter_idx+1)*self.prompt_bsz]
            ctx = self.ctx[batch_idx]
            pos = self.pos[batch_idx]

            self.iter_idx += 1
            if self.iter_idx == self.n_iter:
                self.iter_idx = 0
        else:
            ctx = self.ctx
            pos = self.pos

        prompt_size = ctx.shape[0]
        tokenized_prompts = self.tokenized_prompts.unsqueeze(1).repeat(1, prompt_size, 1).view(self.n_cls*prompt_size, -1)

        n_cls = self.n_cls

        ctx_end = ctx[pos==2]
        n_end = ctx_end.shape[0]
        prefix = self.token_prefix.unsqueeze(1).repeat(1, n_end, 1, 1)
        suffix = self.token_suffix.unsqueeze(1).repeat(1, n_end, 1, 1)
        ctx_end = ctx_end.unsqueeze(0).repeat(n_cls, 1, 1, 1)
        prompts_end = torch.cat([prefix, ctx_end, suffix], dim=2)

        ctx_middle = ctx[pos==1]
        n_middle = ctx_middle.shape[0]
        prompts_middle = []
        half_n_ctx = self.n_ctx // 2
        for i in range(n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_middle, 1, 1)
            ctx_i_half1 = ctx_middle[:, :half_n_ctx, :].unsqueeze(0)
            ctx_i_half2 = ctx_middle[:, half_n_ctx:, :].unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_middle, 1, dim)
                ctx_i_half1, # (1, n_middle, n_ctx//2, dim)
                class_i, # (1, n_middle, name_len, dim)
                ctx_i_half2, # (1, n_middle, n_ctx//2, dim)
                suffix_i # (1, n_middle, *, dim)
            ], dim=2)
            prompts_middle.append(prompt)
        prompts_middle = torch.cat(prompts_middle, dim=0)

        ctx_front = ctx[pos==0]
        n_front = ctx_front.shape[0]
        prompts_front = []
        for i in range(self.n_cls):
            name_len = self.name_lens[i]
            prefix_i = self.token_prefix[i:i+1, :, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            class_i = self.token_suffix[i:i+1, :name_len, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            suffix_i = self.token_suffix[i:i+1, name_len:, :].unsqueeze(1).repeat(1, n_front, 1, 1)
            ctx_i = ctx_front.unsqueeze(0)
            prompt = torch.cat([
                prefix_i, # (1, n_front, 1, dim)
                class_i, # (1, n_front, name_len, dim)
                ctx_i, # (1, n_front, n_ctx, dim)
                suffix_i # (1, n_front, *, dim)
            ], dim=2)
            prompts_front.append(prompt)
        prompts_front = torch.cat(prompts_front, dim=0)

        prompts = torch.cat([prompts_end,prompts_middle, prompts_front], dim=1).view(prompt_size*n_cls, -1, self.ctx_dim)

        if infer:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.ctx
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CustomCLIP(nn.Module):
    def __init__(self, cfg, class_names, clip_model):
        super().__init__()
        n_ctx = cfg.TRAINER.ProDA.N_CTX
        n_prompt = cfg.TRAINER.ProDA.N_PROMPT
        prompt_bsz = cfg.DATALOADER.TRAIN_X.BATCH_SIZE
        multi_label = cfg.DATALOADER.MULTILABEL
        self.n_class = len(class_names)
        self.n_prompt = n_prompt

        # text enoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.prompt_learner = PromptLearner(cfg, class_names, clip_model, n_ctx=n_ctx, n_prompt=n_prompt,
                                            prompt_bsz=prompt_bsz)

        # image encoder
        self.image_encoder = clip_model.visual

        self.logit_scale = clip_model.logit_scale

        self.multi_label = multi_label

    def forward(self, image, labels=None, training=False):

        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        n_class = self.n_class

        if not training:
            text_features = self.text_features
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()
            return logits

        else:
            # assert labels is not None
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner()
            n_prompt = text_prompt.shape[0] // n_class

            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(n_class, n_prompt, -1)
            text_mean = text_features.mean(dim=1)

            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_mean.t()

            if not self.multi_label:
                batch_size = labels.shape[0]

                text_features = text_features - text_mean.unsqueeze(1)
                diag_cov_martix = text_features.permute(2, 0, 1) @ text_features.permute(2, 1, 0)
                diag_cov_martix /= n_prompt + 1
                refined_logits = torch.einsum("bd, dik -> bik", [image_features ** 2, diag_cov_martix])

                sigma = refined_logits[torch.arange(batch_size), labels, labels].unsqueeze(-1) + \
                        refined_logits[:, torch.arange(n_class), torch.arange(n_class)] - \
                        2 * refined_logits[torch.arange(batch_size), labels, :]

                logits += 0.5 * (logit_scale ** 2) * sigma.view(-1, n_class)

            loss_m = None
            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            loss_m = dis[~torch.eye(self.n_prompt, dtype=torch.bool, device='cuda')].abs().mean()

            return logits, loss_m, F.cross_entropy(logits, labels)

    @torch.no_grad()
    def set_classifier(self):
        text_prompt, tokenized_prompts = self.prompt_learner(infer=True)
        try:
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
        except:
            text_features = []
            batch_size = 1000
            for bi in range(text_prompt.shape[0] // batch_size):
                batch_text_features = self.text_encoder(text_prompt[bi * 1000:(bi + 1) * 1000],
                                                        tokenized_prompts[bi * 1000:(bi + 1) * 1000])
                text_features.append(batch_text_features)
            text_features = torch.cat(text_features, dim=0)
        n_dim = text_features.shape[-1]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(self.n_class, self.n_prompt, -1)
        text_features = text_features.mean(dim=1)
        self.text_features = text_features

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype


@TRAINER_REGISTRY.register()
class ProDA(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.ProDA.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ProDA.PREC == "fp32" or cfg.TRAINER.ProDA.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            # print(name, param.requires_grad)
            if name_to_update not in name:
                # Make sure that ProDA prompts are updated
                if "ProDA" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ProDA.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.ProDA.PREC
        if prec == "amp":
            with autocast():
                logits, loss_m, loss = model(image, label, training=True)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            logits, loss_m, loss = model(image, label, training=True)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)


