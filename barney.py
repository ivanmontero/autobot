
from fairseq.models.autoencoder import Autoencoder
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers import PretrainedConfig
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

class BarneyConfig(PretrainedConfig):
    model_type = "barney"

    def __init__(self, fairseq_args, num_labels, classification_dropout, **kwargs):
        self.fairseq_args = vars(fairseq_args) if not isinstance(fairseq_args, dict) else fairseq_args
        super().__init__(**PretrainedConfig.get_config_dict(self.fairseq_args["huggingface_model"])[0])
        self.huggingface_model = self.fairseq_args["huggingface_model"]
        self.classification_dropout = classification_dropout
        self.num_labels = num_labels

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path, **kwargs):
        super_config_dict, super_kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
        super_config_dict.update(kwargs)
        return super_config_dict, super_kwargs

class Barney(PreTrainedModel):  # Sequence Classification
    config_class = BarneyConfig

    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.bert = None  # TODO
        self.bottleneck = None  # TODO
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.classification_dropout)
        self.classifier = nn.Linear(config.fairseq_args["encoder_embed_dim"], config.num_labels)
    
    def set_num_labels(self, num_labels):
        self.num_labels = num_labels
        self.classifier = nn.Linear(config.fairseq_args["encoder_embed_dim"], num_labels)

    @staticmethod
    def from_fairseq(filename, num_labels=1, dropout=0.1):
        autoencoder = Autoencoder.from_pretrained(
            filename,
            checkpoint_file='checkpoint_best.pt').models[0]
        return Barney.from_fairseq_autoencoder(autoencoder, num_labels, dropout)

    @staticmethod
    def from_fairseq_autoencoder(autoencoder, num_labels=1, classification_dropout=0.1):
        barney = Barney(BarneyConfig(autoencoder.args, num_labels, classification_dropout))
        barney.bert = autoencoder.encoder.model
        barney.bottleneck = autoencoder.encoder.bottleneck
        for p in barney.parameters():  # Ensure all parameters are trainable
            p.requires_grad = True
        return barney

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        x = outputs[0].transpose(0, 1)

        if self.bottleneck is not None:
            bottleneck_out = self.bottleneck(x[0,:,:].unsqueeze(0), x[1:,:,:], x[1:,:,:], key_padding_mask=(attention_mask[:,1:] == False) if attention_mask is not None else None)[0].squeeze(0)
        else:
            bottleneck_out = x[0,:,:]

        pooled_output = self.dropout(bottleneck_out)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits, outputs.hidden_states, outputs.attentions

    def forward_embedding(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        x = outputs[0].transpose(0, 1)

        bottleneck_out = self.bottleneck(x[0,:,:].unsqueeze(0), x[1:,:,:], x[1:,:,:], key_padding_mask=(attention_mask[:,1:] == False) if attention_mask is not None else None)[0].squeeze(0)

        return bottleneck_out

        # return SequenceClassifierOutput(
        #     loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
        # )


# class BertPreTrainedModel(PreTrainedModel):
#     """ An abstract class to handle weights initialization and
#         a simple interface for downloading and loading pretrained models.
#     """

#     config_class = BertConfig
#     load_tf_weights = load_tf_weights_in_bert
#     base_model_prefix = "bert"
#     authorized_missing_keys = [r"position_ids"]


# @add_start_docstrings(
#     """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
#     the pooled output) e.g. for GLUE tasks. """,
#     BERT_START_DOCSTRING,
# )
# class BertForSequenceClassification(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.num_labels = config.num_labels

#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)

#         self.init_weights()

#     @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
#     @add_code_sample_docstrings(
#         tokenizer_class=_TOKENIZER_FOR_DOC,
#         checkpoint="bert-base-uncased",
#         output_type=SequenceClassifierOutput,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#         position_ids=None,
#         head_mask=None,
#         inputs_embeds=None,
#         labels=None,
#         output_attentions=None,
#         output_hidden_states=None,
#         return_dict=None,
#     ):
#         r"""
#         labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
#             Labels for computing the sequence classification/regression loss.
#             Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
#             If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
#             If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )

#         pooled_output = outputs[1]

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)

#         loss = None
#         if labels is not None:
#             if self.num_labels == 1:
#                 #  We are doing regression
#                 loss_fct = MSELoss()
#                 loss = loss_fct(logits.view(-1), labels.view(-1))
#             else:
#                 loss_fct = CrossEntropyLoss()
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

#         if not return_dict:
#             output = (logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return SequenceClassifierOutput(
#             loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions,
#         )



    # def forward(self, src_tokens, src_lengths, return_all_hiddens: bool = False):
    #     """
    #     Args:
    #         src_tokens (LongTensor): tokens in the source language of shape
    #             `(batch, src_len)`
    #         src_lengths (torch.LongTensor): lengths of each source sentence of
    #             shape `(batch)`
    #         return_all_hiddens (bool, optional): also return all of the
    #             intermediate hidden states (default: False).
    #     Returns:
    #         namedtuple:
    #             - **encoder_out** (Tensor): the last encoder layer's output of
    #               shape `(src_len, batch, embed_dim)`
    #             - **encoder_padding_mask** (ByteTensor): the positions of
    #               padding elements of shape `(batch, src_len)`
    #             - **encoder_embedding** (Tensor): the (scaled) embedding lookup
    #               of shape `(batch, src_len, embed_dim)`
    #             - **encoder_states** (List[Tensor]): all intermediate
    #               hidden states of shape `(src_len, batch, embed_dim)`.
    #               Only populated if *return_all_hiddens* is True.
    #     """

    #     # # compute padding mask
    #     attention_mask = src_tokens != self.padding_idx

    #     x, pooler_output, hidden_states = self.model(input_ids=src_tokens, attention_mask=attention_mask.float(), output_hidden_states=True)

    #     # masked_logits = self.model_for_mlm.cls(x)[0].transpose(0, 1)

    #     x = x.transpose(0, 1)

    #     bottleneck_out = self.bottleneck(x[0,:,:].unsqueeze(0), x[1:,:,:], x[1:,:,:], key_padding_mask=(attention_mask[:,1:] == False) if attention_mask is not None else None)[0].squeeze(0)

    #     return AutoencoderEncoderOut(
    #         encoder_out=x,  # T x B x C
    #         encoder_padding_mask=(attention_mask == False),  # B x T
    #         src_tokens=None,
    #         src_lengths=None,
    #         bottleneck_out=bottleneck_out,  # B x C
    #     )