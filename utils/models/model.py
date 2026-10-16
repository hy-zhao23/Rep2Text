from transformers import AutoModelForCausalLM, AutoConfig, LlamaConfig, LlamaModel, LlamaForCausalLM 
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from .einv_arch import EInvMetaModel, EInvMetaForCausalLM
from torch import nn
import torch as th
from typing import Optional, List, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from dataclasses import dataclass

@dataclass
class EInvCausalLMOutputWithPast(CausalLMOutputWithPast):
    # Projected vectors in token-embedding space (B, T, D)
    proj: Optional[th.FloatTensor] = None

class EInvConfig(LlamaConfig):
    model_type = "einv_llama"

class EInvLlamaModel(EInvMetaModel, LlamaModel):
    config_class = EInvConfig

    def __init__(self, config: LlamaConfig):
        super(EInvLlamaModel, self).__init__(config)

class EInvLlamaForCausalLM(EInvMetaForCausalLM, LlamaForCausalLM):
    config_class = EInvConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = EInvLlamaModel(config)
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)  # how tensor is split across GPUs
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model
    
    def get_lm_head(self):
        return self.lm_head
    
    def forward(
        self,
        input_ids: th.LongTensor = None,
        attention_mask: Optional[th.Tensor] = None,
        position_ids: Optional[th.LongTensor] = None,
        past_key_values: Optional[List[th.FloatTensor]] = None,
        inputs_embeds: Optional[th.FloatTensor] = None,
        labels: Optional[th.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        hidden_embeds: Optional[th.FloatTensor] = None,
        return_dict: Optional[bool] = True,
        # ✅ NEW: accept cache_position and future kwargs from generate()
        cache_position: Optional[th.LongTensor] = None
    ) -> EInvCausalLMOutputWithPast:

        # if inputs_embeds is not provided, we need to prepare the inputs for einv. But in generation and finetuning, upstream should provide the inputs_embeds.
        if inputs_embeds is None:
            inputs_embeds, attention_mask = self.get_input_embeds_einv(input_ids, attention_mask, hidden_embeds)

        # ---- NEW: compute projector output from the provided hidden_embeds, if any ----
        proj = None
        if hidden_embeds is not None and hasattr(self.model, "embedding_projector"):
            # keep projector math stable in fp32; gradients still flow
            proj = self.model.embedding_projector(hidden_embeds)
        
        # If inputs_embeds is None (e.g. continuation step in cached generation
        # where only 1 new token id is fed), fall back to input_ids so that
        # the base model has at least one of them.
        base_kwargs = dict(
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
            shift_labels=labels,
            use_cache=use_cache,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        if inputs_embeds is not None:
            base_kwargs["inputs_embeds"] = inputs_embeds
        else:
            base_kwargs["input_ids"] = input_ids
        base_out: CausalLMOutputWithPast = super().forward(**base_kwargs)

        return EInvCausalLMOutputWithPast(
            loss=base_out.loss,
            logits=base_out.logits,
            past_key_values=base_out.past_key_values,
            hidden_states=base_out.hidden_states,
            attentions=base_out.attentions,
            proj=proj
        )

    @th.no_grad()
    def generate(
        self,
        input_ids: Optional[th.Tensor] = None,
        hidden_embeds: Optional[th.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, th.LongTensor]:
        return super().generate(
            input_ids=input_ids,
            hidden_embeds=hidden_embeds,
            **kwargs,
        )

AutoConfig.register("einv_llama", EInvConfig)
AutoModelForCausalLM.register(EInvConfig, EInvLlamaForCausalLM)