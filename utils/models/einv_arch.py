from abc import ABC, abstractmethod

from .latent_mlp import EmbeddingProjector
import torch
from .constants import *
from transformers import AddedToken, AutoConfig
from utils import p_args, logger

class EInvMetaModel:
    def __init__(self, config):
        super(EInvMetaModel, self).__init__(config)

        self.config = config
        if hasattr(config, 'hidden_size'):
            emb_cfg = AutoConfig.from_pretrained(p_args.model.emb_id)
            emb_hidden = getattr(emb_cfg, "hidden_size", None)
            if emb_hidden is None:
                raise ValueError("emb_id hidden_size not found; cannot build embedding projector")
            self.embedding_projector = EmbeddingProjector(emb_hidden, config.hidden_size)
        else:
            self.embedding_projector = None
            raise ValueError("embedding_projector is not initialized")

    def get_embedding_projector(self):
        return self.embedding_projector

class EInvMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_input_embeds_einv(
        self, input_ids, attention_mask, hidden_embeds
    ):
        if hidden_embeds is None or input_ids.shape[1] == 1:
            return None, attention_mask


        # 1) Set attention mask based on labels if not provided
        if attention_mask is None:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None:
                attention_mask = (input_ids != pad_id)
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        # 2) rebuild input embeddings
        new_embeds = []
        _hidden_embs = self.get_model().embedding_projector(hidden_embeds)
        emb_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_EM_TOKEN)
        tok_emb = self.get_input_embeddings()

        for b_idx, curIDs in enumerate(input_ids):
            emb_pos = (curIDs == emb_id).nonzero(as_tuple=True)[0]
            if emb_pos.numel() == 0:
                new_embeds.append(tok_emb(curIDs))
                continue
            
            s_pos = emb_pos[0].item()
            e_pos = emb_pos[-1].item() + 1  # exclusive
            assert torch.equal(emb_pos, torch.arange(s_pos, e_pos, device=emb_pos.device)), \
                "Multiple non-contiguous <embedding> blocks are not supported."
            
            # num_embeds = e_pos - s_pos

            l_ids, r_ids = curIDs[:s_pos], curIDs[e_pos:]
            left_emb  = tok_emb(l_ids)  if l_ids.numel()  else None
            right_emb = tok_emb(r_ids) if r_ids.numel() else None

            # Ensure hidden_embeds[batch_idx] has the right shape for concatenation
            hid_emb = _hidden_embs[b_idx]
            if hid_emb.dim() == 1:
                hid_emb = hid_emb.unsqueeze(0)

            assert hid_emb.size(0) == (e_pos - s_pos), \
                f"Projector produced {hid_emb.size(0)} vecs, but found {(e_pos-s_pos)} <embedding> tokens."

            # 👇 Compose new embeddings (align devices for sharded device_map="auto")
            ref_dev = (left_emb if left_emb is not None else (right_emb if right_emb is not None else hid_emb)).device
            parts = []
            if left_emb is not None:  parts.append(left_emb.to(ref_dev))
            parts.append(hid_emb.to(ref_dev))
            if right_emb is not None: parts.append(right_emb.to(ref_dev))
            new_embeds.append(torch.cat(parts, dim=0))

        # Align all rows to the first row's device before stacking
        target_dev = new_embeds[0].device
        new_embeds = [e.to(target_dev) for e in new_embeds]
        input_embeds = torch.stack(new_embeds, dim=0)

        return input_embeds, attention_mask
    
    def initialize_einv_tokenizer(self, tokenizer):
        # Store the tokenizer for later use
        self.tokenizer = tokenizer

        # 1) Build special tokens list using AddedToken to prevent normalization/stripping
        desired_specials = [
            AddedToken(DEFAULT_EM_TOKEN, special=True, normalized=False),
            # AddedToken(DEFAULT_PLACEHOLDER_TOKEN, special=True, normalized=False),
            # AddedToken("[INST]", special=True, normalized=False),
            # AddedToken("[/INST]", special=True, normalized=False),
            # AddedToken("<<SYS>>", special=True, normalized=False),
            # AddedToken("<</SYS>>", special=True, normalized=False),
        ]

        # 1a) Merge with existing additional_special_tokens without duplicates
        existing = list(tokenizer.additional_special_tokens) or []
        existing_set = set(existing)
        merged = existing[:]  # keep order
        for t in desired_specials:
            if str(t) not in existing_set:
                merged.append(t)
                existing_set.add(str(t))

        # 1b) Apply specials (counts only *new* ones)
        num_new_tokens = tokenizer.add_special_tokens({"additional_special_tokens": merged})

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
                num_new_tokens += 1
        
        in_emb = self.get_input_embeddings() ## nn.Embedding
        out_emb = self.get_output_embeddings() ## often lm_head
        
        # 2) Resize embeddings to new vocab size
        if num_new_tokens > 0:
            self.resize_token_embeddings(len(tokenizer), mean_resizing=False)

            with torch.no_grad():
                in_w = in_emb.weight

                # 3) Initialize new token rows to mean of old rows            
                input_avg = in_w[:-num_new_tokens].mean(dim=0, keepdim=True)
                in_w[-num_new_tokens:].copy_(input_avg)

                if out_emb is not None and hasattr(out_emb, "weight") and out_emb.weight.shape[0] == len(tokenizer):
                    out_w = out_emb.weight
                    output_avg = out_w[:-num_new_tokens].mean(dim=0, keepdim=True)
                    out_w[-num_new_tokens:].copy_(output_avg)

        # 4）re-tie to keep input/output consistent
        # This will copy input embeddings to lm_head if the model ties them.
        if hasattr(self, "tie_weights"):
            try:
                self.tie_weights()
            except Exception:
                # Some architectures don’t implement tying; ignore.
                pass

        if p_args.proj.use_patch_token:
            for p in in_emb.parameters():
                p.requires_grad = False
            if out_emb is not None:
                for p in out_emb.parameters():
                    p.requires_grad = False

        self.tokenizer = tokenizer  # save for later use
