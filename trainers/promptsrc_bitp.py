import copy
import os.path as osp
import numpy as np
import math
from functools import reduce
from operator import mul
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .imagenet_templates import IMAGENET_TEMPLATES

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    connect_method = "joint"
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    weight_threshold = cfg.TRAINER.BMIP.WEIGHT_THRESHOLD
    if not zero_shot_model:
        design_details = {"trainer": 'PROMPTSRC',
                          "vision_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_VISION,
                          "language_depth": cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT, "vision_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_VISION * 2,
                          "language_ctx": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT * 2,
                          "bitp_length": cfg.TRAINER.PROMPTSRC.N_CTX_TEXT * 2,
                          "connect_method": connect_method,
                          "weight_threshold": weight_threshold}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0,
                          "weight_threshold": weight_threshold}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # 输出compound_prompts_deeper_text的shape
        # print(compound_prompts_deeper_text.shape) 
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        combined = [x, compound_prompts_deeper_text, 0]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class VLPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # Make sure Language depth >= 1
        assert cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH_TEXT >= 1, "In Independent VL prompting, Language prompt depth should be >=1" \
                                                        "\nPlease use VPT trainer if you want to learn only vision " \
                                                        "branch"
        n_ctx = cfg.TRAINER.PROMPTSRC.N_CTX_TEXT
        ctx_init = cfg.TRAINER.PROMPTSRC.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        ctx_v_dim = clip_model.visual.class_embedding.shape[0]
        val_v = math.sqrt(6. / float(3 * reduce(mul, [16], 1) + ctx_v_dim))  # noqa
        val_t = math.sqrt(6. / float(3 * reduce(mul, [16], 1) + ctx_dim))  # noqa
        self.compound_prompts_depth = cfg.TRAINER.PROMPTSRC.PROMPT_DEPTH
        self.device = cfg.DEVICE
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if cfg.TRAINER.PROMPTSRC.CONNECT_METHOD != "joint":
            if ctx_init and (n_ctx) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
        else:
            if ctx_init and (n_ctx * 2) <= 4:
                # use given words to initialize context vectors
                ctx_init = ctx_init.replace("_", " ")
                prompt = clip.tokenize(ctx_init)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(prompt).type(dtype)
                ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
                prompt_prefix = ctx_init
            else:
                # random initialization
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * (n_ctx * 2))
        
        print(f'Initial text context: "{prompt_prefix}"')
        print(self.compound_prompts_depth)
        print(f"Number of context words (tokens) for Language prompting: {n_ctx}")
        print(f"Number of context words (tokens) for Vision prompting: {cfg.TRAINER.PROMPTSRC.N_CTX_VISION}")
        
        print("begin make prompts")
        self.ctx = nn.Parameter(ctx_vectors)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        # Also create frozen CLIP
        clip_model_temp = load_clip_to_cpu(cfg, True).float().to(self.device)
        clip_model_temp_image = load_clip_to_cpu(cfg, True)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            self.ZS_image_encoder = clip_model_temp_image.visual
            # Now pre-compute the frozen VL embeddings
            all_teacher_features = []
            # Using multiple text templates to ensure textual diversity during training
            for single_template in IMAGENET_TEMPLATES:
                x = [single_template.replace("{}", name) for name in classnames]
                x_tokenized = torch.cat([clip.tokenize(p) for p in x])
                text_features = clip_model_temp.encode_text(x_tokenized.to(self.device))
                all_teacher_features.append(text_features.unsqueeze(1))
        self.fixed_embeddings = torch.cat(all_teacher_features, dim=1).mean(dim=1)

        self.proj_t = nn.Linear(ctx_dim, ctx_v_dim)
        self.proj_v = nn.Linear(ctx_v_dim, ctx_dim)
        self.proj_t.half()
        self.proj_v.half()
        self.ctx_t = nn.Parameter(ctx_vectors)
        # Visual prompts are 768, as they are projected from 512
        self.ctx_v = nn.Parameter(torch.empty(embedding.shape[1], ctx_v_dim))
        nn.init.normal_(self.ctx_v, std=0.02)
        self.ctx_prompts_visual = self.ctx_v[:n_ctx, :]
        # Minimum can be 1, which defaults to shallow BITP
        # compound prompts
        # Visual prompts are 768, as they are projected from 512
        if cfg.SHARE.SHARE_PARAMETER_VISUAL:
            self.compound_prompts_visual = nn.ParameterList([self.ctx_prompts_visual
                                                             for _ in range(self.compound_prompts_depth - 1)])
            self.single_layer_v = nn.Linear(ctx_v_dim, ctx_dim)
        else:
            self.compound_prompts_visual = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_v_dim))
                                                             for _ in range(self.compound_prompts_depth - 1)])

            for single_para in self.compound_prompts_visual:
                # nn.init.normal_(single_para, std=0.02)
                nn.init.uniform_(single_para, -val_v, val_v)

            single_layer_v = nn.Linear(ctx_v_dim, ctx_dim)
            self.compound_prompt_projections_v = _get_clones(single_layer_v, self.compound_prompts_depth - 1)

        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)

        if cfg.SHARE.SHARE_PARAMETER_TEXT:
            self.single_layer_t = nn.Linear(ctx_dim, ctx_v_dim)
        else:
            # Also make corresponding projection layers, for each prompt
            single_layer_t = nn.Linear(ctx_dim, ctx_v_dim)
            self.compound_prompt_projections_t = _get_clones(single_layer_t, self.compound_prompts_depth - 1)
        self.share_para_visual = cfg.SHARE.SHARE_PARAMETER_VISUAL
        self.share_para_text = cfg.SHARE.SHARE_PARAMETER_TEXT

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        print("end of construct_prompts")

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx_t = self.ctx_t.to(torch.float16)
        ctx_v = self.ctx_v[:self.n_ctx, :].to(torch.float16)
        # print(ctx_t.shape)
        # print(ctx_v.shape)

        if ctx_t.dim() == 2:
            ctx_t = ctx_t.unsqueeze(0).expand(self.n_cls, -1, -1)
            # ctx_v = ctx_v.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx_t, prefix, suffix)

        # Before returning, need to transform
        # prompts to 768 for the visual side
        visual_deep_prompts = []
        if self.share_para_text:
            for index in range(self.compound_prompts_depth - 1):
                visual_deep_prompts.append(self.single_layer_t(self.compound_prompts_text[index]))  # 768
        else:
            for index, layer in enumerate(self.compound_prompt_projections_t):
                visual_deep_prompts.append(layer(self.compound_prompts_text[index]))  # 768
        # visual_deep_prompts = visual_deep_prompts.to(torch.float16)

        text_deep_prompts = []

        if self.share_para_visual:
            for index in range(self.compound_prompts_depth - 1):
                text_deep_prompts.append(self.single_layer_v(self.compound_prompts_visual[index]))  # 512
        else:
            for index, layer in enumerate(self.compound_prompt_projections_v):
                text_deep_prompts.append(layer(self.compound_prompts_visual[index]))  # 512
        text_deep_prompts.append(ctx_t)
        # text_deep_prompts = text_deep_prompts.to(torch.float16)
        # Now the other way around
        # We will project the textual prompts from 512 to 768
        shared_ctx_v = self.proj_v(ctx_v)
        return prompts, self.proj_t(
            self.ctx_t), self.compound_prompts_text, visual_deep_prompts, ctx_v, shared_ctx_v, self.compound_prompts_visual, text_deep_prompts  # pass here original, as for visual 768 is required



class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = VLPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.total_ctx = self.prompt_learner.n_ctx * 2
        self.n_cls = len(classnames)
        self.connect_method = cfg.TRAINER.PROMPTSRC.CONNECT_METHOD
        self.device = cfg.DEVICE


    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text_t, deep_compound_prompts_vision_t, prompts_v, shared_ctx_v, deep_compound_prompts_vision_v, deep_compound_prompts_text_v = self.prompt_learner()
        deep_compound_prompts_text = []
        if self.connect_method == "atten":
            list_deep_compound_prompts_text_t = list(deep_compound_prompts_text_t)
            for i in range(len(list_deep_compound_prompts_text_t)):
                deep_compound_prompts_text.append(Atten(deep_compound_prompts_text_v[i],
                                                        list_deep_compound_prompts_text_t[i])
                                                  + list_deep_compound_prompts_text_t[i])
        elif self.connect_method == "joint":
            list_deep_compound_prompts_text_t = list(deep_compound_prompts_text_t)
            for i in range(len(list_deep_compound_prompts_text_t)):
                deep_compound_prompts_text.append(torch.stack([deep_compound_prompts_text_v[i],
                                                               list_deep_compound_prompts_text_t[i]], dim=0).view(self.total_ctx, 512))
        elif self.connect_method == "add":
            list_deep_compound_prompts_text_t = list(deep_compound_prompts_text_t)
            deep_compound_prompts_text = list_deep_compound_prompts_text_t + deep_compound_prompts_text_v
        else:
            print("error connect_method")
            print("please choose add, atten or joint")

        deep_compound_prompts_vision = []
        if self.connect_method == "atten":
            list_deep_compound_prompts_vision_v = list(deep_compound_prompts_vision_v)
            # deep_compound_prompts_vision = list_deep_compound_prompts_vision_v + deep_compound_prompts_vision_t
            for i in range(len(list_deep_compound_prompts_vision_v)):
                deep_compound_prompts_vision.append(Atten(deep_compound_prompts_vision_t[i],
                                                        list_deep_compound_prompts_vision_v[i])
                                                    + list_deep_compound_prompts_vision_v[i])
            shared_ctx = Atten(shared_ctx, prompts_v) + prompts_v

        elif self.connect_method == "joint":
            list_deep_compound_prompts_vision_v = list(deep_compound_prompts_vision_v)
            for i in range(len(list_deep_compound_prompts_vision_v)):
                deep_compound_prompts_vision.append(
                    torch.stack([deep_compound_prompts_vision_v[i],
                                 deep_compound_prompts_vision_t[i]], dim=0).view(self.total_ctx, prompts_v.shape[1]))
            # deep_compound_prompts_vision = torch.Tensor(deep_compound_prompts_vision)
            shared_ctx = torch.stack([shared_ctx, prompts_v], dim=0).view(self.total_ctx, prompts_v.shape[1])

        elif self.connect_method == "add":
            list_deep_compound_prompts_vision_v = list(deep_compound_prompts_vision_v)
            deep_compound_prompts_vision = list_deep_compound_prompts_vision_v + deep_compound_prompts_vision_t
            shared_ctx = shared_ctx + prompts_v

        else:
            print("error connect_method")
            print("please choose add, atten or joint")
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        # print(image.type(self.dtype))
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Compute the prompted logits
        logits = logit_scale * image_features @ text_features.t()
        if self.prompt_learner.training:
            # Now calculate the frozen pre-trained features
            fixed_embeddings = self.prompt_learner.fixed_embeddings  # precomputed pre-trained frozen textual features
            fixed_embeddings = fixed_embeddings / fixed_embeddings.norm(dim=-1, keepdim=True)
            with torch.no_grad():
                zero_shot_features = self.prompt_learner.ZS_image_encoder(image.type(self.dtype))
                zero_shot_features = zero_shot_features / zero_shot_features.norm(dim=-1, keepdim=True)
                # Compute pre-trained frozen visual features
                zero_shot_logits = logit_scale * zero_shot_features.to(self.device) @ fixed_embeddings.half().to(self.device).t()

            return F.cross_entropy(logits,
                                   label), text_features, fixed_embeddings, zero_shot_features, \
                   image_features, zero_shot_logits, logits
        else:
            return logits


def _get_clones(module, N):
    # 深度复制，不会共享
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@TRAINER_REGISTRY.register()
class PromptSRC_BITP(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.PROMPTSRC.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        self.device = cfg.DEVICE
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PROMPTSRC.PREC == "fp32" or cfg.TRAINER.PROMPTSRC.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            else:
                if "ZS_image_encoder" in name:
                    param.requires_grad_(False)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        print(f"Parameters count: {len(enabled)}")
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1
        N = cfg.OPTIM.MAX_EPOCH
        mean = cfg.TRAINER.PROMPTSRC.GPA_MEAN
        stdev = cfg.TRAINER.PROMPTSRC.GPA_STD
        gauss = self.get_gauss(mean, stdev)
        self.gauss = np.array([gauss(a) for a in range(1, N + 1)])
        self.gauss = self.gauss / sum(self.gauss)
        self.scaler = GradScaler() if cfg.TRAINER.PROMPTSRC.PREC == "amp" else None
        # Keep model with GPA
        self.previous_model_gpa = None

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler
        self.device = self.cfg.DEVICE

        prec = self.cfg.TRAINER.PROMPTSRC.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss_ce, normalized_text_features, zs_clip_text_embeddings, zs_image_embedd, image_ft, \
            zero_shot_logits, logits = model(image, label)
            # Calculate the L_SCL_text loss
            loss_scl_text = F.l1_loss(normalized_text_features, zs_clip_text_embeddings.to(self.device),
                                      reduction='mean') * self.cfg.TRAINER.PROMPTSRC.TEXT_LOSS_WEIGHT
            # Calculate the L_SCL_image loss
            loss_scl_image = F.l1_loss(image_ft, zs_image_embedd.to(self.device),
                                       reduction='mean') * self.cfg.TRAINER.PROMPTSRC.IMAGE_LOSS_WEIGHT
            # Now calculate L_SCL_logits
            L_SCL_logits = F.kl_div(
                F.log_softmax(logits / 1, dim=1),
                F.log_softmax(zero_shot_logits / 1, dim=1),
                reduction='sum',
                log_target=True
            ) * (1 * 1) / logits.numel()
            L_SCL = (L_SCL_logits + loss_scl_text + loss_scl_image)
            loss = (loss_ce + L_SCL)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            # Means one epoch is completed, perform GPA
            self.step_counter = self.step_counter + 1
            current_epoch_weight = self.gauss[self.step_counter - 2]
            current_model_weights = copy.deepcopy(model.state_dict())
            weighted_state_dict = self.state_dict_weighting(current_model_weights, current_epoch_weight)
            if self.previous_model_gpa is None:
                self.previous_model_gpa = weighted_state_dict
            else:
                self.previous_model_gpa = self.state_dict_add(weighted_state_dict, self.previous_model_gpa)

        if self.step_counter == self.model.total_epochs + 1:
            print("Using GPA model for final inference...")
            model.load_state_dict(self.previous_model_gpa)
            self.model.load_state_dict(self.previous_model_gpa)
        return loss_summary

    def state_dict_weighting(self, main_dict, weightage, prompt_only=False):
        # Average all parameters
        updated_dict = copy.deepcopy(main_dict)
        if not prompt_only:
            for key in main_dict:
                updated_dict[key] = main_dict[key] * weightage
            return updated_dict
        else:
            return main_dict * weightage

    def state_dict_add(self, dict1, dict2, prompt_only=False):
        # Average all parameters
        if not prompt_only:
            modified_dict = dict2
            for key in dict1:
                modified_dict[key] = (modified_dict[key] + dict1[key])
            return modified_dict
        else:
            return dict1 + dict2

    def get_gauss(self, mu, sigma):
        gauss = lambda x: (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return gauss

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