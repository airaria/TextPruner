import torch
from torch import nn
import os

from torch.nn.functional import softmax, log_softmax
from .utils import move_to_device, generate_mask, infer_model_type
from .utils import infer_logits, infer_loss
from ..configurations import  TransformerPruningConfig, GeneralConfig

from ..model_map import MODEL_MAP
import logging
from tqdm import tqdm
from collections import abc
from typing import Mapping, Optional
from copy import deepcopy
logger = logging.getLogger(__name__)

class TransformerPruner:
    '''
    Args:
        model : The model to be pruned.
        transformer_pruning_config : a :class:`~textpruner.configurations.TransformerPruningConfig` object.
        general_config : a :class:`~textpruner.configurations.GeneralConfig` object.
        base_model_prefix : The prefix of the base model, i.e., the name of the base model as a member in the model. \
For example, if ``model.bert_encoder = BertModel(...)``, then the ``base_model_prefix`` is ``bert_encoder``. \
TextPruner will infer the ``base_model_prefix`` so we can leave its value as ``None``. But if it fails, users have to set its value explicitly.
    '''
    def __init__(self, model : nn.Module, 
                       transformer_pruning_config : Optional[TransformerPruningConfig] = None,
                       general_config : Optional[GeneralConfig] = None,
                       base_model_prefix : Optional[str] = None):
        self.model = model
        base_model, model_type = infer_model_type(model, base_model_prefix)
        assert model_type in MODEL_MAP, \
            f"Model type {model_type} is not supported, or not understood. Model type must be one of {list(MODEL_MAP.keys())}"
        self.base_model = base_model
        self.model_type = model_type
        self.model_structure = MODEL_MAP[self.model_type]['structure']

        self.general_config = GeneralConfig() if general_config is None else general_config
        self.transformer_pruning_config = TransformerPruningConfig() if transformer_pruning_config is None else transformer_pruning_config

        self.model.to(self.general_config.device)

        self.output_dir : str = self.general_config.output_dir

        # None before pruning
        self.head_mask : Optional[torch.Tensor] = None
        self.ffn_mask : Optional[torch.Tensor] = None
        self.keep_shape : Optional[bool] = None
        os.makedirs(self.output_dir, exist_ok=True)

        self.shoule_cache_logits = True
        self.soft_labels = []
        if self.transformer_pruning_config.use_logits is True:
            self.model_rep = deepcopy(model)
            self.model_rep.half().to(model.device)
        self.save_dir = None

    def prune(self, dataloader=None, adaptor=None, batch_postprocessor=None, 
                head_mask: Optional[torch.Tensor] =None, ffn_mask: Optional[torch.Tensor]=None, 
                keep_shape=False, save_model=True, rewrite_cache=True):
        '''
        Prunes the transformers. If ``self.transformer_pruning_config.pruning_method=='masks'``, the pruner prune the attention heads and the FFN neurons based on the 
        ``head_masks`` and ``ffn_masks``; if ``self.transformer_pruning_config.pruning_method=='iterative'``, the pruner prune the attention heads and the FFN neurons
        based on the importance scores calculated on the batches from the ``dataloader``.

        Args:
            dataloader : a dataloader that generates batches. Each batch should contains both the inputs and the labels.
            adaptor : a function that takes the model output and return the loss.
            batch_postprocessor : a function that takes the batch produced by the dataloader and return a batch. It is used for post-processing the batches if needed.
            head_mask : a tensor of shape ``(num_layers, num_attention_heads)``.  `1` means to keep, `0` means to prune.
            ffn_mask : a tensor of shape ``(num_layers, intermediate_hidden_size)``.  `1` means to keep, `0` means to prune.
            keep_shape : if ``True``, the model is no actually pruned and the model stucture is not changed, but the weights that *should be pruned* are set to zero.
            save_model : whether to save the model when the pruning is finished.
        '''

        pruning_method = self.transformer_pruning_config.pruning_method
        if pruning_method == 'masks':
            if head_mask is not None or ffn_mask is not None:
                save_dir = self.prune_with_masks(head_mask=head_mask, ffn_mask=ffn_mask, set_masks=True, save_model=save_model)
            else:
                raise TypeError("Pruning method is 'masks', but no masks are given.")
        elif pruning_method == 'iterative':
            assert (dataloader is not None ), "Pruning method is 'iterative', but dataloader is not given."
            save_dir = self.iterative_pruning(dataloader, adaptor, batch_postprocessor, keep_shape, save_model=save_model, rewrite_cache=rewrite_cache)
        else:
            raise NotImplementedError(f"Unknow pruning method {pruning_method}.")
        self.save_dir = save_dir
        return save_dir

    def prune_with_masks(self,head_mask: Optional[torch.Tensor] = None, 
                                ffn_mask: Optional[torch.Tensor] = None, 
                                keep_shape : bool = False, 
                                set_masks = False, 
                                save_model = False) -> Optional[str]:
        if head_mask is None:
            head_mask = self.head_mask
        if ffn_mask is None:
            ffn_mask = self.ffn_mask
        if set_masks is True:
            if head_mask is not None:
                self.head_mask = head_mask
            if ffn_mask is not None:
                self.ffn_mask = ffn_mask

        if ffn_mask is not None:
            ffn_mask_tensor = ffn_mask.clone().detach().to(dtype=torch.float32, device=self.general_config.device)
            self.reorder_ffn_weights(ffn_mask_tensor, keep_shape)
        if head_mask is not None:
            if keep_shape:
                head_mask_tensor = head_mask.clone().detach().to(dtype=torch.float32, device=self.general_config.device)
                self.reorder_attention_heads(head_mask_tensor, keep_shape)
            else:
                heads_to_prune_dict = {}
                for layer_num, layer_head in enumerate(head_mask.tolist()):
                    heads_to_prune_dict[layer_num] = []
                    for head_idx, v in enumerate(layer_head):
                        if v==0:
                            heads_to_prune_dict[layer_num].append(head_idx)
                self.base_model.prune_heads(heads_to_prune_dict)
        self.keep_shape = keep_shape
        if save_model is True:
            return self.save_model()

    def iterative_pruning(self, dataloader, adaptor, batch_postprocessor=None, keep_shape=False, save_model=True, rewrite_cache=False) -> Optional[str]:

        target_ffn_size = self.transformer_pruning_config.target_ffn_size
        target_num_of_heads = self.transformer_pruning_config.target_num_of_heads
        n_iters = self.transformer_pruning_config.n_iters
        multiple_of = self.transformer_pruning_config.multiple_of
        head_even_masking = self.transformer_pruning_config.head_even_masking
        ffn_even_masking = self.transformer_pruning_config.ffn_even_masking
        pruning_order = self.transformer_pruning_config.pruning_order

        head_importance_fn = os.path.join(self.output_dir, f'head_importance.pt')
        ffn_importance_fn = os.path.join(self.output_dir,f'ffn_importance.pt')

        if os.path.exists(head_importance_fn) and os.path.exists(ffn_importance_fn) and rewrite_cache is False:
            logger.info(f"Loading pre-cached head importance score {head_importance_fn}")
            head_importance = torch.load(head_importance_fn)
            logger.info(f"Loading pre-cached ffn importance score {ffn_importance_fn}")
            ffn_importance = torch.load(ffn_importance_fn)
        else:
            logger.info("Calculating head importance and ffn importance")
            if self.transformer_pruning_config.use_logits:
                head_importance, ffn_importance = self.get_importance_score_with_logits(dataloader, adaptor, batch_postprocessor)
            else:
                head_importance, ffn_importance = self.get_importance_score(dataloader, adaptor, batch_postprocessor)
            head_importance = head_importance.cpu() # (num_layers, num_heads)
            ffn_importance = ffn_importance.cpu() # (num_layers, intermediate_size)
            # Save importance score
            logger.info("Save...")
            torch.save(head_importance, head_importance_fn)
            torch.save(ffn_importance, ffn_importance_fn)

        total_num_of_heads = head_importance.size(0)*head_importance.size(1)
        total_ffn_size = ffn_importance.size(0)*ffn_importance.size(1)

        total_target_ffn_size = target_ffn_size * ffn_importance.size(0)
        total_target_num_of_heads = target_num_of_heads *head_importance.size(0)


        ffn_size_per_iter = (total_ffn_size - total_target_ffn_size) // n_iters
        num_of_heads_per_iter = (total_num_of_heads - total_target_num_of_heads) // n_iters
        ffn_size_res = (total_ffn_size - total_target_ffn_size) % n_iters
        num_of_heads_res = (total_num_of_heads - total_target_num_of_heads) % n_iters

        dffn_size = total_ffn_size
        dnum_of_heads = total_num_of_heads

        if pruning_order is None:
            for i in range(n_iters):
                logger.info(f'Number of pruning iterations: {i+1}/{n_iters}')
                if i > 0:
                    logger.info("Calculating head importance and ffn importance")
                    if self.transformer_pruning_config.use_logits:
                        head_importance, ffn_importance = self.get_importance_score_with_logits(dataloader, adaptor, batch_postprocessor)
                    else:
                        head_importance, ffn_importance = self.get_importance_score(dataloader, adaptor, batch_postprocessor)
                    head_importance = head_importance.cpu() # (num_layers, num_heads)
                    ffn_importance = ffn_importance.cpu() # (num_layers, intermediate_size)

                    assert torch.all(head_importance==head_importance*self.head_mask)
                    assert torch.all(ffn_importance==ffn_importance*self.ffn_mask)
                    #head_importance *= self.head_mask
                    #ffn_importance *= self.ffn_mask

                dffn_size -= ffn_size_per_iter + 1 if i < ffn_size_res else ffn_size_per_iter
                dnum_of_heads -= num_of_heads_per_iter + 1 if i < num_of_heads_res else num_of_heads_per_iter

                self.head_mask = generate_mask(head_importance, dnum_of_heads, head_even_masking)
                self.ffn_mask = generate_mask(ffn_importance, dffn_size, ffn_even_masking, multiple_of=multiple_of)

                logger.info(f"New ffn size:{self.ffn_mask.sum(-1).tolist()}")
                logger.info(f"New num heads:{self.head_mask.sum(-1).tolist()}")

                if i==n_iters-1:
                    self.prune_with_masks(keep_shape=keep_shape, save_model=False)
                else:
                    self.prune_with_masks(keep_shape=True, save_model=False)
        else:
            for i in range(n_iters * 2):  # n_iters for head, n_iters for ffn
                logger.info(f'Number of pruning iterations: {i+1}/{n_iters * 2}')
                if pruning_order=='head-first':
                    current_is_head = (i%2==0)
                    current_is_ffn = (i%2==1)
                else:
                    current_is_ffn = (i%2==0)
                    current_is_head = (i%2==1)
                if i > 0:
                    logger.info("Calculating head importance and ffn importance")
                    if self.transformer_pruning_config.use_logits:
                        head_importance, ffn_importance = self.get_importance_score_with_logits(dataloader, adaptor, batch_postprocessor)
                    else:
                        head_importance, ffn_importance = self.get_importance_score(dataloader, adaptor, batch_postprocessor)
                    head_importance = head_importance.cpu() # (num_layers, num_heads)
                    ffn_importance = ffn_importance.cpu() # (num_layers, intermediate_size)


                if current_is_ffn:
                    dffn_size -= ffn_size_per_iter + 1 if i//2 < ffn_size_res else ffn_size_per_iter
                    self.ffn_mask = generate_mask(ffn_importance, dffn_size, ffn_even_masking, multiple_of=multiple_of)
                    logger.info(f"New ffn size:{self.ffn_mask.sum(-1).tolist()}")
                if current_is_head:
                    dnum_of_heads -= num_of_heads_per_iter + 1 if i//2 < num_of_heads_res else num_of_heads_per_iter
                    self.head_mask = generate_mask(head_importance, dnum_of_heads, head_even_masking)
                    logger.info(f"New num heads:{self.head_mask.sum(-1).tolist()}")

                if i==2 * n_iters-1:
                    self.prune_with_masks(keep_shape=keep_shape, save_model=False)
                else:
                    self.prune_with_masks(keep_shape=True, save_model=False)

        #clear cache
        self.soft_labels = []
        self.shoule_cache_logits = True

        logger.info("Head and ffn masks have been generated, can be accessed via self.head_mask and self.ffn_mask")
        if save_model is True:
            return self.save_model()


    def save_masks(self,name='mask.pt') -> str:
        save_dir = os.path.join(self.general_config.output_dir,f'head_ffn_masks')
        os.makedirs(save_dir, exist_ok=True)
        torch.save((self.head_mask,self.ffn_mask),os.path.join(save_dir,f'{name}'))
        # save config
        logger.info(f"Masks have been saved to {save_dir}")

        return save_dir


    def save_model(self, dir_name=None) -> str:
        ffn_sizes = self.ffn_mask.to(int).sum(-1).tolist()
        if self.keep_shape is False:
            ffn_size = ffn_sizes[0]
            num_of_heads = self.head_mask.sum().item() / self.head_mask.size(0) # self.head_mask.to(int).sum().item()
            if len(set(ffn_sizes)) != 1:
                raise NotImplementedError("Cannot save pruned model with different ffn size per layer with keep_shape=False. \
Call TransformerPruner.save_masks or TransformerPruner.save_jit_model manually instead.")
            else:
                self.base_model.config.intermediate_size = ffn_size
        else:
            ffn_size = self.ffn_mask.size(1) #base_model.config.intermediate_size
            num_of_heads = self.head_mask.size(1) #self.transformer_pruning_config.target_num_of_heads

        if dir_name is None:
            save_dir = os.path.join(self.general_config.output_dir,f'pruned_H{num_of_heads}F{ffn_size}')
        else:
            save_dir = os.path.join(self.general_config.output_dir,dir_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(),os.path.join(save_dir,'pytorch_model.bin'))
        # save config
        self.base_model.config.save_pretrained(save_dir)
        logger.info(f"Model and configuration have been saved to {save_dir}")

        return save_dir


    def save_jit_model(self, example_inputs, dir_name=None) -> str:
        self.model.eval()
        with torch.no_grad():
            traced_model = torch.jit.trace(self.model, example_inputs=example_inputs, strict=False)
        if dir_name is None:
            save_dir = os.path.join(self.general_config.output_dir,'pruned_H{num_of_heads}F{ffn_size}_traced')
        else:
            save_dir = os.path.join(self.general_config.output_dir,dir_name)
        os.makedirs(save_dir, exist_ok=True)
        torch.jit.save(traced_model, os.path.join(save_dir,'pytorch_model.ts'))

        return save_dir

    def reorder_attention_heads(self, head_mask, keep_shape = False):

        n_layers = head_mask.size(0)
        head_size = int(self.base_model.config.hidden_size / self.base_model.config.num_attention_heads)

        #assert torch.all(new_num_heads_vec==new_num_heads_vec[0]), "Numbers of heads in each layer must be equal"
        
        att_queries = self.model_structure.get_att_query(self.base_model, ignore_model_prefix=True)
        att_keys = self.model_structure.get_att_key(self.base_model, ignore_model_prefix=True)
        att_values = self.model_structure.get_att_value(self.base_model, ignore_model_prefix=True)
        att_outputs = self.model_structure.get_att_output(self.base_model, ignore_model_prefix=True)
        
        for layer_num in range(n_layers):
            query_weight = att_queries[layer_num].weight
            query_bias = att_queries[layer_num].bias
            key_weight = att_keys[layer_num].weight
            key_bias = att_keys[layer_num].bias
            value_weight = att_values[layer_num].weight
            value_bias = att_values[layer_num].bias
            output_weight = att_outputs[layer_num].weight

            # sort query, key, value based on the scores
            query_weight, query_bias = rearange_weights(query_weight,query_bias,head_mask[layer_num],head_size,keep_shape)
            att_queries[layer_num].weight = torch.nn.Parameter(query_weight.contiguous())
            att_queries[layer_num].bias = torch.nn.Parameter(query_bias.contiguous())
            key_weight, key_bias = rearange_weights(key_weight,key_bias,head_mask[layer_num],head_size,keep_shape)
            att_keys[layer_num].weight = torch.nn.Parameter(key_weight.contiguous())
            att_keys[layer_num].bias = torch.nn.Parameter(key_bias.contiguous())
            value_weight, value_bias = rearange_weights(value_weight,value_bias,head_mask[layer_num],head_size,keep_shape)
            att_values[layer_num].weight = torch.nn.Parameter(value_weight.contiguous())
            att_values[layer_num].bias = torch.nn.Parameter(value_bias.contiguous())

            output_weight, _ = rearange_weights(output_weight.transpose(0,1), None, head_mask[layer_num],head_size,keep_shape)
            output_weight = output_weight.transpose(0,1)
            att_outputs[layer_num].weight = torch.nn.Parameter(output_weight.contiguous())


    def reorder_ffn_weights(self, ffn_mask, keep_shape = False):

        head_size = 1 #int(base_model.config.hidden_size / base_model.config.num_attention_heads)

        n_layers = ffn_mask.size(0)        
        ffn_interm = self.model_structure.get_ffn_interm(self.base_model, ignore_model_prefix=True)
        ffn_output = self.model_structure.get_ffn_output(self.base_model, ignore_model_prefix=True)

        for layer_num in range(n_layers):

            inter_weight = ffn_interm[layer_num].weight
            inter_bias = ffn_interm[layer_num].bias
            output_weight = ffn_output[layer_num].weight

            # sort query, key, value based on the confidence scores
            inter_weight, inter_bias = rearange_weights(inter_weight, inter_bias, ffn_mask[layer_num], head_size, keep_shape)
            ffn_interm[layer_num].weight = torch.nn.Parameter(inter_weight)
            ffn_interm[layer_num].bias = torch.nn.Parameter(inter_bias)

            output_weight, _ = rearange_weights(output_weight.transpose(0,1), None, ffn_mask[layer_num], head_size, keep_shape)
            output_weight = output_weight.transpose(0,1).contiguous()
            ffn_output[layer_num].weight = torch.nn.Parameter(output_weight)


    def get_importance_score(self, dataloader,
                                adaptor=None, batch_postprocessor=None) -> torch.Tensor :
        model = self.model

        n_layers = self.model_structure.get_num_layers(self.base_model, ignore_model_prefix=True)
        n_heads = self.base_model.config.num_attention_heads
        intermediate_size = self.base_model.config.intermediate_size

        device = self.general_config.device

        logger.info("***** Running Forward and Backward to calcuate importance score*****")
        logger.info(" Length of dataloader = %d", len(dataloader))
        model.eval()

        head_importance = torch.zeros(n_layers, n_heads).to(device)

        #get ffn weights and bias
        ffn_inter_weights = []
        ffn_inter_biases = []
        ffn_output_weights = []
        att_output_weights = []

        ffn_interm = self.model_structure.get_ffn_interm(self.base_model, ignore_model_prefix=True)
        ffn_output = self.model_structure.get_ffn_output(self.base_model, ignore_model_prefix=True)
        att_output = self.model_structure.get_att_output(self.base_model, ignore_model_prefix=True)
        for layer_num in range(n_layers):
                ffn_inter_weights.append(ffn_interm[layer_num].weight) #.detach().to(device)
                ffn_inter_biases.append(ffn_interm[layer_num].bias) #.detach().to(device)
                ffn_output_weights.append(ffn_output[layer_num].weight) #.detach().to(device)
                att_output_weights.append(att_output[layer_num].weight)

        ffn_importance = torch.zeros(n_layers, intermediate_size).to(device) #ex. (12,3072)
        num_examples = 0.0

        for batch in tqdm(dataloader, desc="Calculating IS with loss"):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            batch = move_to_device(batch, device)
            if isinstance(batch,abc.Mapping):
                outputs = model(**batch)
                batch_num_examples = len(list(batch.values())[0])
            else:
                outputs = model(*batch)
                batch_num_examples = len(batch[0])
            loss = infer_loss(outputs, adaptor)
            loss.backward()

            for layer_num in range(n_layers):
                weight = att_output_weights[layer_num]
                head_importance[layer_num] += (weight.grad * weight).view(weight.size(0),n_heads, -1).sum(dim=(0,2)).abs().detach()  # (num_heads, )

            for layer_num in range(n_layers):
                weight1 = ffn_inter_weights[layer_num]
                bias1 = ffn_inter_biases[layer_num]
                weight2 = ffn_output_weights[layer_num]
                if self.transformer_pruning_config.ffn_even_masking:
                    ffn_importance[layer_num] += ((weight1.grad * weight1).sum(dim=1)+ bias1.grad * bias1).abs().detach()
                ffn_importance[layer_num] += ((weight2.grad * weight2).sum(dim=0)).abs().detach()

            model.zero_grad()
            num_examples += batch_num_examples

        head_importance /= num_examples
        ffn_importance /= num_examples

        return head_importance, ffn_importance



    def get_importance_score_with_logits(self, dataloader,
                                adaptor=None, batch_postprocessor=None) -> torch.Tensor :
        model = self.model

        n_layers = self.model_structure.get_num_layers(self.base_model, ignore_model_prefix=True)
        n_heads = self.base_model.config.num_attention_heads
        intermediate_size = self.base_model.config.intermediate_size

        device = self.general_config.device

        logger.info("***** Running Forward and Backward to calcuate importance score*****")
        logger.info(" Length of dataloader = %d", len(dataloader))
        model.eval()
        self.model_rep.eval()
        head_importance = torch.zeros(n_layers, n_heads).to(device)

        #get ffn weights and bias
        ffn_inter_weights = []
        ffn_inter_biases = []
        ffn_output_weights = []
        att_output_weights = []

        ffn_interm = self.model_structure.get_ffn_interm(self.base_model, ignore_model_prefix=True)
        ffn_output = self.model_structure.get_ffn_output(self.base_model, ignore_model_prefix=True)
        att_output = self.model_structure.get_att_output(self.base_model, ignore_model_prefix=True)
        for layer_num in range(n_layers):
                ffn_inter_weights.append(ffn_interm[layer_num].weight) #.detach().to(device)
                ffn_inter_biases.append(ffn_interm[layer_num].bias) #.detach().to(device)
                ffn_output_weights.append(ffn_output[layer_num].weight) #.detach().to(device)
                att_output_weights.append(att_output[layer_num].weight)

        ffn_importance = torch.zeros(n_layers, intermediate_size).to(device) #ex. (12,3072)
        num_examples = 0.0


        for idx,batch in enumerate(tqdm(dataloader, desc="Calculating IS with logits")):
            if batch_postprocessor is not None:
                batch = batch_postprocessor(batch)
            batch = move_to_device(batch, device)
            if isinstance(batch,abc.Mapping):
                outputs = model(**batch)
                batch_num_examples = len(list(batch.values())[0])
            else:
                outputs = model(*batch)
                batch_num_examples = len(batch[0])

            with torch.no_grad():
                outputs_rep = self.model_rep(**batch) if isinstance(batch,abc.Mapping) else self.model_rep(*batch)
            logits_rep = infer_logits(outputs_rep, adaptor)

            logits = infer_logits(outputs, adaptor)
            #if self.shoule_cache_logits is True: # cache soft labels if the cache is empty
            #    p = softmax(logits, dim=-1).detach()
            #    self.soft_labels.append(p)

            if isinstance(logits,(list,tuple)):
                entropy = 0
                for logits_p, logits_q in zip(logits_rep, logits):
                    current_p = softmax(logits_p, dim=-1).detach()
                    current_q = logits_q
                    entropy += -(log_softmax(current_q,dim=-1) * current_p).sum(dim=-1).mean()
            else:
                current_p = softmax(logits_rep, dim=-1).detach() #p = softmax(logits, dim=-1).detach() #self.soft_labels[idx]
                #current_p = self.soft_labels[idx]
                current_q = logits
                entropy = - (log_softmax(current_q,dim=-1) * current_p).sum(dim=-1).mean()
            entropy.backward()


            for layer_num in range(n_layers):
                weight = att_output_weights[layer_num]
                head_importance[layer_num] += (weight.grad * weight).view(weight.size(0),n_heads, -1).sum(dim=(0,2)).abs().detach()  # (num_heads, )

            for layer_num in range(n_layers):
                weight1 = ffn_inter_weights[layer_num]
                bias1 = ffn_inter_biases[layer_num]
                weight2 = ffn_output_weights[layer_num]
                if self.transformer_pruning_config.ffn_even_masking:
                    ffn_importance[layer_num] += ((weight1.grad * weight1).sum(dim=1)+ bias1.grad * bias1).abs().detach()
                ffn_importance[layer_num] += ((weight2.grad * weight2).sum(dim=0)).abs().detach()

            model.zero_grad()
            num_examples += batch_num_examples

        if self.shoule_cache_logits is True:
            self.shoule_cache_logits = False

        head_importance /= num_examples
        ffn_importance /= num_examples

        return head_importance, ffn_importance


def rearange_weights(weight, bias, mask, head_size, keep_shape = False):
    num_heads = mask.size(0)
    mask_dim3 = mask.view(num_heads,1,1).to(torch.bool) # 12,1,1 ?
    weight_dim3 = weight.view(num_heads,head_size,weight.size(1)) # 12,64,768
    if keep_shape is False:
        selected_weight = weight_dim3.masked_select(mask_dim3)
        new_num_heads = int(mask.sum().item())
    else:
        selected_weight = torch.mul(weight_dim3, mask_dim3)
        new_num_heads = num_heads

    ##reshape back
    selected_weight = selected_weight.view(new_num_heads*head_size, weight.size(1)).contiguous()

    selected_bias = None
    if bias is not None:
        mask_dim2 = mask.view(num_heads,1).to(torch.bool) # 12,1 ?
        bias_dim2 = bias.view(num_heads,head_size) #12,64
        if keep_shape == False:
            selected_bias = bias_dim2.masked_select(mask_dim2)
        else:
            selected_bias = torch.mul(bias_dim2, mask_dim2)
        selected_bias = selected_bias.view(new_num_heads*head_size).contiguous()

    return selected_weight, selected_bias

