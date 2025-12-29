import logging
from dataclasses import fields
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.auto import AutoModelForCausalLM, AutoModelForSequenceClassification

from olmo.config import ModelConfig
from olmo.model import OLMo
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .configuration_olmo import OLMoConfig

log = logging.getLogger(__name__)


def create_model_config_from_pretrained_config(config: OLMoConfig, is_cls = False):
    """
    Utility function
    """
    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)
    # add num_labels for being compatible with the AutoSeqClassification downstream task
    model_config = ModelConfig(**kwargs)
    if is_cls:
        num_labels = len(getattr(config,'label2id'))
        # print(f"{config}")
        return model_config, num_labels
    return model_config


class OLMoForCausalLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = OLMoConfig
    base_model_prefix = "model"
    _no_split_modules = ["OLMoBlock"]

    def __init__(self, config: OLMoConfig, model: Optional[OLMo] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = OLMo(model_config, init_params=init_params)
        else:
            self.model = model
        self.word_embeddings = self.model.transformer.wte
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,  # Added parameter
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        ######
        # Create attention bias only if it's not provided for bidirectional finetuning
        # Should only uncomment when performing MNTP finetuning
        ######
        # if attention_bias is None:
        #     seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        #     attention_bias = self.get_bidirectional_attention_bias(seq_len=seq_len, device=input_ids.device)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.embedding_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def get_bidirectional_attention_bias(self, seq_len: int, device: torch.device):
        """
        Create a bidirectional attention bias for full sequence attention.
        The bias matrix will not restrict attention in any direction.
        """
        # Bias shape: (1, 1, seq_len, seq_len)
        bias = torch.zeros(1, 1, seq_len, seq_len, device=device)
        return bias

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    # TODO: these are required to make the implementation complete.
    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     pass
    #
    # def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
    #     pass
    #
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     pass

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        """
        This function is intentionally left as a no-op.

        Weight tying is handled as follows:
        - When the model is initialized, the `ff_out` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.

        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.embedding_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        Note:
            This method differs from the base class implementation by resizing the `embedding_size` attribute of the
            model configuration instead of the `vocab_size`. It also includes a warning if the resized `embedding_size`
            is less than the `vocab_size`. In OLMo, `embedding_size` refers to the dimensionality of the model's token
            embeddings, while `vocab_size` refers to the number of unique tokens in the vocabulary.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.embedding_size = model_embeds.weight.shape[0]
        self.model.config.embedding_size = model_embeds.weight.shape[0]

        # Check if the embedding size is less than the vocab size
        if self.config.embedding_size < self.config.vocab_size:
            warning_message = (
                f"Resizing token embeddings to size {self.config.embedding_size}, which is less than the vocab size "
                f"{self.config.vocab_size} defined in the model configuration. Make sure your tokenizer's vocabulary "
                "size is less than or equal to the new token embedding size."
            )
            log.warning(warning_message)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModelForCausalLM.register(OLMoConfig, OLMoForCausalLM)


from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
class OLMoForSequenceCLS(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = OLMoConfig
    base_model_prefix = "model"
    _no_split_modules = ["OLMoBlock"]

    def __init__(self, config: OLMoConfig, model: Optional[OLMo] = None, init_params: bool = False):
        super().__init__(config)
        if not model:
            model_config,num_labels = create_model_config_from_pretrained_config(config,is_cls=True)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = OLMo(model_config, init_params=init_params)
        else:
            self.model = model
        self.word_embeddings = self.model.transformer.wte
        self.num_labels = num_labels
        print(f"num_labels: {self.num_labels}")
        self.score = torch.nn.Linear(config.hidden_size, self.num_labels, bias=False)


        ###############
        # mix resolution head
        ################
        # self.CNN = CNN_Head(output_size=self.num_labels,cnn_output_dim=config.hidden_size, kernel_sizes=[4,9],dropout_rate=0.11,
        #                     num_cnn_layers=2)
    def get_bidirectional_attention_bias(self, seq_len: int, device: torch.device):
        """
        Create a bidirectional attention bias for full sequence attention.
        The bias matrix will not restrict attention in any direction.
        """
        # Bias shape: (1, 1, seq_len, seq_len)
        bias = torch.zeros(1, 1, seq_len, seq_len, device=device)
        return bias
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")
        ######
        # Create attention bias only if it's not provided
        ######
        # if attention_bias is None:
        #     seq_len = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        #     attention_bias = self.get_bidirectional_attention_bias(seq_len=seq_len, device=input_ids.device)
        ######
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ########
        # The output_hidden_states flag is set as the output format of olmo is the following:
        # return OLMoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)
        # so we have to forcely set the output hidden_states flag
        ########
        output_hidden_states = True
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = outputs.hidden_states[-1]
        # assume that the padding is done by prepadding at the left of the input sequence
        # the logit of the last non-padding token is logit[:,-1,:]
        logits = self.score(hidden_states)
        ##########
        seq_lengths = attention_mask.sum(dim=-1)
        # instead of taking the mean, we can also take the last token, taking the length of the sequence
        pooled_logits = torch.stack(
                [
                    logits[i, length - 1, :]
                    for i, length in enumerate(seq_lengths)
                ],
                dim=0,
            )
        ##########
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (pooled_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )
    def forward_new(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        onehot: Optional[torch.Tensor] = None, # New field
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[
            Cache
        ] = None,  # This is a hack mitigation of an issue in transformers `4.39.x` https://github.com/huggingface/transformers/issues/29426
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in OLMo")
        ######
        # input_ids shape
        ######

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        ########
        # The output_hidden_states flag is set as the output format of olmo is the following:
        # return OLMoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)
        # so we have to forcely set the output hidden_states flag
        ########
        output_hidden_states = True
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        #----------
        # outputs = self.model.forward(
        #     input_ids=input_ids,
        #     input_embeddings=inputs_embeds,
        #     attention_mask=attention_mask,
        #     attention_bias=attention_bias,
        #     past_key_values=past_key_values,
        #     use_cache=use_cache,
        #     output_hidden_states=output_hidden_states,
        # )
        # hidden_states = outputs.hidden_states[-1]
        #-------------
        # assume that the padding is done by prepadding at the left of the input sequence
        # the logit of the last non-padding token is logit[:,-1,:]
        # logits = self.score(hidden_states)
        # pooled_logits = hidden_states[:,-1,:]
        pooled_logits = self.CNN(onehot)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
            # output = (pooled_logits,) + outputs[1:] #------
            # return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            # past_key_values=outputs.attn_key_values,
            # hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    # TODO: these are required to make the implementation complete.
    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     pass
    #
    # def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
    #     pass
    #
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     pass

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: torch.nn.Module):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: torch.nn.Module):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        """
        This function is intentionally left as a no-op.

        Weight tying is handled as follows:
        - When the model is initialized, the `ff_out` layer is conditionally defined based on the `weight_tying` configuration.
        See: `if not config.weight_tying: self.transformer.update(...)` in `olmo/model.py`.
        - When computing logits, the `wte` weights are used directly if `weight_tying` is enabled.
        See: `if self.config.weight_tying: logits = F.linear(x, self.transformer.wte.weight, None)` in the `forward` method.

        Therefore, there is no need to explicitly tie the weights in this function.
        """
        pass

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        """
        Resizes input token embeddings matrix of the model if `new_num_tokens != config.embedding_size`.

        Takes care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The new number of tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the embedding matrix to a multiple of the provided value. If `new_num_tokens` is set to
                `None` will just pad the embedding to a multiple of `pad_to_multiple_of`.

                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128. For more
                details about this, or help on choosing the correct value for resizing, refer to this guide:
                https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#requirements-tc

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.

        Note:
            This method differs from the base class implementation by resizing the `embedding_size` attribute of the
            model configuration instead of the `vocab_size`. It also includes a warning if the resized `embedding_size`
            is less than the `vocab_size`. In OLMo, `embedding_size` refers to the dimensionality of the model's token
            embeddings, while `vocab_size` refers to the number of unique tokens in the vocabulary.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        if new_num_tokens is None and pad_to_multiple_of is None:
            return model_embeds

        # Update base model and current model config
        self.config.embedding_size = model_embeds.weight.shape[0]
        self.model.config.embedding_size = model_embeds.weight.shape[0]

        # Check if the embedding size is less than the vocab size
        if self.config.embedding_size < self.config.vocab_size:
            warning_message = (
                f"Resizing token embeddings to size {self.config.embedding_size}, which is less than the vocab size "
                f"{self.config.vocab_size} defined in the model configuration. Make sure your tokenizer's vocabulary "
                "size is less than or equal to the new token embedding size."
            )
            log.warning(warning_message)

        # Tie weights again if needed
        self.tie_weights()

        return model_embeds
# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModelForSequenceClassification.register(OLMoConfig, OLMoForSequenceCLS)
