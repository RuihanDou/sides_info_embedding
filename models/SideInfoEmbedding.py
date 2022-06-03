from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'configure.conf'))
os.chdir(conf.get_string('work_path'))

from data_process.side_information import get_side_info_tensor, get_side_info_mask
from data_process.skip_gram_negative_sampling import *
import tensorflow as tf


class SideInfoEmbedding(tf.keras.Model):
    def __init__(self, side_info_size: int = conf.get_int('side_info_tag_size')
                 , embedding_dim: int = conf.get_int('embedding_dim')
                 , side_info_indices_tensor=get_side_info_tensor()
                 , side_info_indices_mask=get_side_info_mask()
                 , layer_name: str = conf.get_string('layer_name')
                 ):
        super(SideInfoEmbedding, self).__init__()
        self.side_info_size = side_info_size
        self.embedding_dim = embedding_dim
        self.side_info_indices_tensor = side_info_indices_tensor
        self.side_info_indices_mask = side_info_indices_mask
        self.side_info_embedding = tf.keras.layers.Embedding(self.side_info_size, self.embedding_dim, name=layer_name)

    def call(self, pair):
        # targets.shape = (batch_size, ) , contexts.shape = (batch_size, )
        targets, contexts = pair
        # targets_side_info_idx.shape = contexts_side_info_idx.shape =(batch_size, side_info_max_num_tags)
        targets_side_info_idx = tf.gather(params=self.side_info_indices_tensor, indices=targets)
        contexts_side_info_idx = tf.gather(params=self.side_info_indices_tensor, indices=contexts)
        # targets_side_info_mask.shape = contexts_side_info_mask.shape =(batch_size, side_info_max_num_tags)
        targets_side_info_mask = tf.gather(params=self.side_info_indices_mask, indices=targets)
        contexts_side_info_mask = tf.gather(params=self.side_info_indices_mask, indices=contexts)
        # targets_embedding.shape = contexts_embedding.shape = (batch_size, side_info_max_num_tags, embedding_dim)
        targets_embedding = self.side_info_embedding(targets_side_info_idx)
        contexts_embedding = self.side_info_embedding(contexts_side_info_idx)
        # masked_targets_embedding.shape = masked_contexts_embedding.shape = (batch_size, side_info_max_num_tags, embedding_dim)
        masked_targets_embedding = tf.einsum('bxy,bx->bxy', targets_embedding, targets_side_info_mask)
        masked_contexts_embedding = tf.einsum('bxy,bx->bxy', contexts_embedding, contexts_side_info_mask)
        # targets_item_embedding.shape = contexts_item_embedding.shape = (batch_size, embedding_dim)
        targets_item_embedding = tf.math.reduce_sum(masked_targets_embedding, 1)
        contexts_item_embedding = tf.math.reduce_sum(masked_contexts_embedding, 1)
        # dots.shape = (batch_size)
        dots = tf.einsum('be,be->b', targets_item_embedding, contexts_item_embedding)
        return dots

if __name__ == '__main__':
    dataset = generate_train_epoch_dataset(batch_size=16, buffer_size=200)
    model = SideInfoEmbedding()
    for pair, label in dataset.take(10):
        predict = model(pair)
        print(predict)
        print(label)
        print("\n")