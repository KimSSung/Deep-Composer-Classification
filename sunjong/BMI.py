import torch
from torch import Tensor
import torch.nn as nn
# import torch.nn.functional as F
# import random
# import numpy as np

from transformers import BertModel
# BertConfig args:
#	[vocab_size, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072,
# 		hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=16, initializer_range=0.02):


class BMI(PreTrainedBertModel):
	""" 'B'ert model for 'MI'di genre classification.
	This module is composed of the BERT model with a linear layer on top of
	the pooled output.

	Params:
		`config`: a BertConfig class instance with the configuration to build a new model.
		`num_labels`: the number of classes for the classifier. Default = 5.

	Inputs:
		`input_ids`: total # 388 * instnum + 3(CLS,SEP,PAD). [batch_size, sequence_length]
					Here each one hot vector of sequence has '388 * instnum + 3' dimension

		`token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length]
					Here, all inputs are type 0 (no question(0) and answer(1) form)

		`attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length]
					For real token(not padding) -> 1, for padded position -> 0

		`labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
			with indices selected in [0, ..., num_labels].

	Outputs: ---------Don't know why used-------------
		if `labels` is not `None`:
			Outputs the CrossEntropy classification loss of the output with the labels.

		if `labels` is `None`:
			Outputs the classification logits of shape [batch_size, num_labels].
	"""

	def __init__(self, config, num_labels=5):
		super(BMI, self).__init__(config)
		self.num_labels = num_labels
		self.bert = BertModel(config)
		self.dropout = nn.Dropout(config.hidden_dropout_prob)
		self.classifier = nn.Linear(config.hidden_size, num_labels)
		self.apply(self.init_bert_weights)

	def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
		# BertModel output: (sequence_output, pooled_output, (option: hidden_states), (option: attention))
		# 1. sequence_output: [batch_size, seq_len, dim]
		# 2. pooled_output: get CLS([batch_size, dim]) then * [dim, dim] = [batch_size, dim]
		# we use pooled_out rather than sequence_output
		_, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
		pooled_output = self.dropout(pooled_output)
		logits = self.classifier(pooled_output)

		''' ---------Don't know why used-------------
		if labels is not None:
			loss = nn.CrossEntropyLoss()
			return loss
		else:
			return logits
		'''

		return logits
		
	def freeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = False
	
	def unfreeze_bert_encoder(self):
		for param in self.bert.parameters():
			param.requires_grad = True