"""Custom models for few-shot learning specific operations."""

import torch
import torch.nn as nn
import transformers
import torch.nn.functional as F
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead

import logging
logger = logging.getLogger(__name__)


class RobertaForPromptFinetuning(RobertaPreTrainedModel):
    _tied_weights_keys = ["lm_head.decoder.weight", "lm_head.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        # These attributes should be assigned once the model is initialized
        self.model_args = None
        self.data_args = None
        self.label_word_list = None

        # For regression
        self.lb = None
        self.ub = None

        # For auto label search.
        self.return_full_softmax = None

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mask_pos=None,
        labels=None,
    ):
        batch_size = input_ids.size(0)

        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()

        # Encode everything
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        # Get <mask> token representation
        sequence_output = outputs[0] # batch_size * seq_len * 1024
        sequence_mask_output = sequence_output[torch.arange(sequence_output.size(0)), mask_pos] # batch * 1024

        # Logits over vocabulary tokens
        prediction_mask_scores = self.lm_head(sequence_mask_output)

        # Exit early and only return mask logits.
        if self.return_full_softmax:
            if labels is not None:
                return torch.zeros(1, out=prediction_mask_scores.new()), prediction_mask_scores
            return prediction_mask_scores

        # Return logits for each label
        logits = []
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:, self.label_word_list[label_id]].unsqueeze(-1))
        logits = torch.cat(logits, -1)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression task
                loss_fct = nn.KLDivLoss(log_target=True)
                labels = torch.stack([1 - (labels.view(-1) - self.lb) / (self.ub - self.lb), (labels.view(-1) - self.lb) / (self.ub - self.lb)], -1)
                loss = loss_fct(logits.view(-1, 2), labels)
            else:
                """
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                """
                ###########################
                loss_fct = nn.KLDivLoss()
                target = F.softmax(torch.ones(self.num_labels), dim=0).cuda()

                logits_log = F.log_softmax(logits.view(-1, logits.size(-1)), dim=1)  
                loss_1 = loss_fct(logits_log, target)
                
                logits_avg = torch.mean(logits.view(-1, logits.size(-1)), 0)
                logits_avg_log = F.log_softmax(logits_avg, dim=-1)  
                loss_2 = loss_fct(logits_avg_log, target)
                loss = (loss_1 + loss_2) * 0.5

                # loss = loss_1

                assert loss >= 0
                ###########################

        output = (logits,)
        if self.num_labels == 1:
            # Regression output
            output = (torch.exp(logits[..., 1].unsqueeze(-1)) * (self.ub - self.lb) + self.lb,)
        return ((loss,) + output) if loss is not None else output
