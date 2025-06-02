import math
import warnings
from typing import List, Optional, Tuple, Union, Callable
import os

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers import AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaPreTrainedModel, LlamaModel, LlamaConfig

logger = logging.get_logger(__name__)

class LlamaResidualPlus(LlamaForCausalLM):

    def __init__(self, model, pretrained_model, config):

        super().__init__(config)

        self.model = model

        self.vocab_size = self.config.vocab_size

        self.pretrained_model = pretrained_model
        self.pretrained_model.eval()

        del self.lm_head

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def to_cuda(self):

        self.model.to("cuda")
        self.pretrained_model.to("cuda")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        
        first_outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        second_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        first_logits = first_outputs.logits

        second_logits = second_outputs.logits

        logits = first_logits + second_logits

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + first_outputs.hidden_states[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def save_pretrained(
        self,
        *args,
        **kwargs,
    ):
        return self.model.save_pretrained(*args, **kwargs)
    
    def prepare_inputs_for_generation(self, *args, **kwargs):

        return self.model.prepare_inputs_for_generation(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = None,
        pretrained_model_name_or_path: str = None,
        **kwargs,
    ) -> PreTrainedModel:

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **kwargs
        )

        pretrained_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        config = model.config

        return cls(model=model, pretrained_model=pretrained_model, config=config)