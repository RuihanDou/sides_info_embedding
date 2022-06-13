from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))
import sys
sys.path.append(conf.get_string('work_path'))
from data_process.side_information_pd import get_side_info_tensor, get_side_info_mask
from data_process.skip_gram_negative_sampling import *
import tensorflow as tf


class SideInfoEmbedding(tf.keras.Model):
    def __init__(self, side_info_size: int = conf.get_int('side_info_tag_size')
                 , embedding_dim: int = conf.get_int('embedding_dim')
                 , side_info_indices_tensor=tf.constant([0])
                 , side_info_indices_mask=tf.constant([0])
                 , layer_name: str = conf.get_string('layer_name')
                 ):
        super(SideInfoEmbedding, self).__init__()
        self.side_info_size = side_info_size
        self.embedding_dim = embedding_dim
        self.side_info_indices_tensor = side_info_indices_tensor
        self.side_info_indices_mask = side_info_indices_mask
        self.side_info_embedding = tf.keras.layers.Embedding(self.side_info_size, self.embedding_dim, name=layer_name)

    def call(self, pair):
        # targets.shape = (batch_size, 1) , contexts.shape = (batch_size, 1)
        targets, contexts = pair
        # # targets.shape = (batch_size, 1) , contexts.shape = (batch_size, 1)
        # targets_side_info_idx.shape = contexts_side_info_idx.shape =(batch_size, 1, side_info_max_num_tags)
        targets_side_info_idx = tf.gather(params=self.side_info_indices_tensor, indices=targets)
        contexts_side_info_idx = tf.gather(params=self.side_info_indices_tensor, indices=contexts)
        # targets_side_info_mask.shape = contexts_side_info_mask.shape =(batch_size, 1, side_info_max_num_tags)
        targets_side_info_mask = tf.gather(params=self.side_info_indices_mask, indices=targets)
        contexts_side_info_mask = tf.gather(params=self.side_info_indices_mask, indices=contexts)
        # targets_embedding.shape = contexts_embedding.shape = (batch_size, 1, side_info_max_num_tags, embedding_dim)
        targets_embedding = self.side_info_embedding(targets_side_info_idx)
        contexts_embedding = self.side_info_embedding(contexts_side_info_idx)
        # masked_targets_embedding.shape = masked_contexts_embedding.shape = (batch_size, 1, side_info_max_num_tags, embedding_dim)
        # b = batch_size l = 1, t = side_info_max_num_tags, e = mbedding_dim
        masked_targets_embedding = tf.einsum('blte,blt->blte', targets_embedding, targets_side_info_mask)
        masked_contexts_embedding = tf.einsum('blte,blt->blte', contexts_embedding, contexts_side_info_mask)
        # targets_item_embedding.shape = contexts_item_embedding.shape = (batch_size, 1, embedding_dim)
        targets_item_embedding = tf.math.reduce_sum(masked_targets_embedding, 2)
        contexts_item_embedding = tf.math.reduce_sum(masked_contexts_embedding, 2)
        # dots.shape = (batch_size, 1)
        dots = tf.einsum('ble,ble->bl', targets_item_embedding, contexts_item_embedding)
        return dots
