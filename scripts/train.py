from pyhocon import ConfigFactory
import os
conf = ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), '..', 'config.conf'))
os.chdir(conf.get_string('work_path'))
import sys
sys.path.append(conf.get_string('work_path'))
from utils.access_tool import *
from utils.tokenizer_tool import *
from data_process.side_information_pd import *
from data_process.asymmetrical_weighted_graph import AsymmetricalWeightedGraph
from data_process.graph_file_pd import sequences_dataframe_to_graph_pair_file
from data_process.random_walk_sequence_generator import RandomWalkSequenceGenerator
from data_process.skip_gram_negative_sampling import generate_train_epoch_dataset
from models.SideInfoEmbedding import SideInfoEmbedding
from utils.plot_tool import plot_auc_trend
from result_process.generate_vecs import extract_side_info_vectors, extract_item_vectors
from datetime import datetime

"""
从 configure 里解析 训练的配置
"""

# side information 的表格存储路径
side_infomation_path = conf.get_string('side_infomation_path')
# 序列表格路径
sequences_path = conf.get_string('sequences_path')
# 读取 keras.Tokenizer 的路径, 该tokenizer 把 side_information 换成 embedding 矢量的查询索引
tokenizer_path = conf.get_string('tokenizer_path')
# item 对应side info
side_info_dict_path = conf.get_string('side_info_dict_path')
# 连续整数空间中id 对应side info
neg_samp_id_to_side_info_dict_path = conf.get_string('neg_samp_id_to_side_info_dict_path')
# 存储图的路径
graph_file_path = conf.get_string('graph_file_path')
# 存储一个epoch随机游走序列路径
walk_sequence_file_path = conf.get_string('walk_sequence_file_path')
# item 转化为 连续整数空间中id 的映射
item_to_neg_samp_id_path = conf.get_string('item_to_neg_samp_id_path')
# 连续整数空间中id 转化为 item 的映射
neg_samp_id_to_item_path = conf.get_string('neg_samp_id_to_item_path')
# 设置本地pysark session配置
excecutor_memory = conf.get_string('excecutor_memory')
driver_memory = conf.get_string('driver_memory')
max_result_size = conf.get_string('max_result_size')
# item 的 tag (side info) 的种类 个数 + [UNK]
side_info_tag_size = conf.get_int('side_info_tag_size')
# 单条item 的 tag (side info) 最多个数 为了方便 219 --> 220
side_info_max_num_tags = conf.get_int('side_info_max_num_tags')
# 每条tag的 embedding size 大小
embedding_dim = conf.get_int('embedding_dim')
# tag 的 category
side_info_category = conf.get_list('side_info_category')
# sequences 的 category
sequence_category = conf.get_string('sequence_category')
# 随机游走生成的最长序列长度
max_walk_seq_length = conf.get_int('max_walk_seq_length')
# 随机游走生成的最短序列长度
min_walk_seq_length = conf.get_int('min_walk_seq_length')
# 终止游走概率
walk_end_probability = conf.get_float('walk_end_probability')
# 图的节点数(方便调试用)
vertices_num = conf.get_int('vertices_num')
window_size = conf.get_int('window_size')
negative_sample_rate = conf.get_float('negative_sample_rate')
# 训练迭代数
epochs = conf.get_int('epochs')
# 训练批大小
batch_size = conf.get_int('batch_size')
# shuffle 用的缓存大小
buffer_size = conf.get_int('buffer_size')
# embedding layer 的名称
layer_name = conf.get_string('layer_name')
# # callbacks存储位置
# callbacks_log = conf.get_string('callbacks_log')
# 存储训练完成的 side_info 的 csv 文件 包含三列 id， tag， vec
side_info_vec_csv_path = conf.get_string('side_info_vec_csv_path')
# 存储训练完成后的 side_info 的 tag -> vec 的dict
side_info_tag_vec_dict_path = conf.get_string('side_info_tag_vec_dict_path')
# 存储训练完成后的 side_info 的 id -> vec 的dict
side_info_id_vec_dict_path = conf.get_string('side_info_id_vec_dict_path')
# 存储训练完成的 item 的 csv 文件 包含三列 id， item， vec
item_vec_csv_path = conf.get_string('item_vec_csv_path')
# 存储训练完成后的 item 的 item -> vec 的dict
item_vec_dict_path = conf.get_string('item_vec_dict_path')
# 存储训练完成后的 item 的 id -> vec 的dict
item_id_vec_dict_path = conf.get_string('item_id_vec_dict_path')


# 数据路径
data_path = conf.get_string('data_path')

# 日志路径
log_path = conf.get_string('log_path')

# 结果路径
rst_path = conf.get_string('rst_path')




if __name__ == '__main__':
    """
    创建训练需要文件夹
    """
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    if not os.path.exists(log_path):
        os.mkdir(log_path)

    if not os.path.exists(rst_path):
        os.mkdir(rst_path)
    """
        数据准备
        """
    """
        第一步  side_info 预处理

        读取 side info 的 csv 文件 存储 dict
        """
    side_info_dict = get_side_information_dict(side_info_path=side_infomation_path
                                               , tokenizer=load_tokenizer(tokenizer_path)
                                               , padding_max_size=side_info_max_num_tags
                                               , catagory_list=side_info_category)
    save_dict(side_info_dict_path, side_info_dict)
    """
    第二步 使用点击序列组成图 生成图文件
    """
    sequences_dataframe_to_graph_pair_file(sequences_path=sequences_path, graph_file_path=graph_file_path)
    """
    第三步 通过图文件，生成图对象, 并且将图作为 随机游走对象 的初始参数构建 随机游走对象
    """
    graph = AsymmetricalWeightedGraph(file_path=graph_file_path)

    random_walk = RandomWalkSequenceGenerator(asymmerical_weighted_graph=graph
                                              , max_walk_seq_length=max_walk_seq_length
                                              , min_walk_seq_length=min_walk_seq_length
                                              , walk_end_probability=walk_end_probability
                                              , walk_sequence_file_path=walk_sequence_file_path
                                              , item_to_neg_samp_id_path=item_to_neg_samp_id_path
                                              , neg_samp_id_to_item_path=neg_samp_id_to_item_path)
    """
    第四步 把 side_info 关联上 处理完的图(经过random walk生成，item有了连续空间的id编码，图中 vertex 数量小于全量 item)

    并 储存
    """
    neg_samp_id_side_info_dict = get_neg_samp_id_side_info_dict(
        item_to_neg_samp_id_dict=load_dict(item_to_neg_samp_id_path), side_info_dict=side_info_dict)
    save_dict(neg_samp_id_to_side_info_dict_path, neg_samp_id_side_info_dict)

    """
    第五步 side info 编码为对应tensor
    """
    side_info_tensor = get_side_info_tensor(neg_samp_id_side_info_dict)
    side_info_mask = get_side_info_mask(neg_samp_id_side_info_dict)

    """
    初始化模型
    """
    embedding_model = SideInfoEmbedding(side_info_size=side_info_tag_size, embedding_dim=embedding_dim,
                                        side_info_indices_tensor=side_info_tensor,
                                        side_info_indices_mask=side_info_mask, layer_name=layer_name)

    embedding_model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                            metrics='accuracy')


    """
    训练开始，生成游走序列，并且skip-gram生成dataset，进行一次fit
    
    日志只进行每 epoch 的记录，设置 tf.summary 对每次迭代进行观察
    
    训练过程 per step 记录 loss, accuracy, 
    验证过程 per epoch 记录 recall， PR-AUC， ROC—AUC
    """
    # 编写摘要记录器
    cur_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join(log_path, '{}-train'.format(cur_time))
    evalu_log_dir = os.path.join(log_path, '{}-evalu'.format(cur_time))
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    evalu_summary_writer = tf.summary.create_file_writer(evalu_log_dir)
    # eval 评价使用的 metrics
    recall_m = tf.keras.metrics.Recall()
    roc_m = tf.keras.metrics.AUC(curve='ROC')
    pr_m = tf.keras.metrics.AUC(curve='PR')
    # 记录全局训练
    globel_step = 0
    for epoch in range(epochs):
        print("epoch {0:03d} begin".format(epoch))
        print("epoch train")
        random_walk.generate_epoch()
        dataset = generate_train_epoch_dataset(walk_sequence_path=walk_sequence_file_path,
                                               item_to_id_dict=load_dict(item_to_neg_samp_id_path),
                                               vertex_num=vertices_num,
                                               window_size=window_size, negative_sample_rate=negative_sample_rate,
                                               batch_size=batch_size, buffer_size=buffer_size)


        print("model fit on dataset:")

        loss_t = tf.constant(0.0)
        accuracy_t = tf.constant(0.0)

        dataset_size = len(dataset)
        for i, ((t, c), l) in tqdm(dataset.enumerate()):
            batch_logs = embedding_model.train_step(((t,c), l))
            # {'loss': <tf.Tensor: shape=(), dtype=float32, numpy=1.3398242>, 'accuracy': <tf.Tensor: shape=(), dtype=float32, numpy=0.53125>}
            if i == dataset_size - 1:
                loss_t = batch_logs['loss']
                accuracy_t = batch_logs['accuracy']
            with train_summary_writer.as_default():
                tf.summary.scalar('train-loss', batch_logs['loss'], globel_step + i)
                tf.summary.scalar('train-accuracy', batch_logs['accuracy'], globel_step + i)
        globel_step += dataset_size
        print("Train summary: loss: {} ,  accuracy: {} \n".format(
            loss_t.numpy(), accuracy_t.numpy()
        ))
        random_walk.clean_epoch()


        """
        迭代内的监控，每个epoch，计算一下新采样的序列skip-gram的pair 结果的AUC(PR 和 ROC)
        """
        print("epoch evaluate")

        recall_m.reset_state()
        roc_m.reset_state()
        pr_m.reset_state()
        random_walk.generate_epoch()
        dataset = generate_train_epoch_dataset(walk_sequence_path=walk_sequence_file_path,
                                               item_to_id_dict=load_dict(item_to_neg_samp_id_path),
                                               vertex_num=vertices_num,
                                               window_size=window_size, negative_sample_rate=negative_sample_rate,
                                               batch_size=batch_size, buffer_size=buffer_size)
        for batch_pair, batch_label in tqdm(dataset):
            groud_truth = tf.squeeze(batch_label)
            prediction = tf.clip_by_value(tf.squeeze(embedding_model(batch_pair)), 0.0, 1.0)
            roc_m.update_state(groud_truth, prediction)
            pr_m.update_state(groud_truth, prediction)
            recall_m.update_state(groud_truth, prediction)


        random_walk.clean_epoch()
        with evalu_summary_writer.as_default():
            tf.summary.scalar('recall', batch_logs['loss'], epoch)
            tf.summary.scalar('PR-AUC', batch_logs['loss'], epoch)
            tf.summary.scalar('ROC-AUC', batch_logs['accuracy'], epoch)

        print("Evaluation summary: recall: {} ,  PR-AUC: {} ,  ROC-AUC: {}\n".format(
            recall_m.result().numpy(),
            pr_m.result().numpy(),
            roc_m.result().numpy()
        ))

        print("epoch save.")

        """
            存储参数
            """
        side_info_df = extract_side_info_vectors(embedding_model=embedding_model,
                                                 tokenizer=load_tokenizer(tokenizer_path),
                                                 csv_path=side_info_vec_csv_path.format(cur_time, epoch),
                                                 tag_dict_path=side_info_tag_vec_dict_path.format(cur_time, epoch),
                                                 id_dict_path=side_info_id_vec_dict_path.format(cur_time, epoch))

        item_df = extract_item_vectors(embedding_model=embedding_model,
                                       item_id_side_info_dict=load_dict(neg_samp_id_to_side_info_dict_path),
                                       id_to_item_dict=load_dict(neg_samp_id_to_item_path),
                                       csv_path=item_vec_csv_path.format(cur_time, epoch),
                                       item_dict_path=item_vec_dict_path.format(cur_time, epoch),
                                       id_dict_path=item_id_vec_dict_path.format(cur_time, epoch))

        print("epoch {0:03d} finished.".format(epoch))

        print("\n")
        print("\n")


    """
    完成训练
    """
    # plot_auc_trend(roc_auc=roc_auc, pr_auc=pr_auc, insert_eval_epoch=insert_eval_epoch)
    # """
    # 存储参数
    # """
    # side_info_df = extract_side_info_vectors(embedding_model=embedding_model, tokenizer=load_tokenizer(tokenizer_path),
    #                                          csv_path=side_info_vec_csv_path, tag_dict_path=side_info_tag_vec_dict_path,
    #                                          id_dict_path=side_info_id_vec_dict_path)
    # item_df = extract_item_vectors(embedding_model=embedding_model,
    #                                item_id_side_info_dict=load_dict(neg_samp_id_to_side_info_dict_path),
    #                                id_to_item_dict=load_dict(neg_samp_id_to_item_path), csv_path=item_vec_csv_path,
    #                                item_dict_path=item_vec_dict_path, id_dict_path=item_id_vec_dict_path)


