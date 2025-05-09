import logging
import re

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import config
import numpy as np

import utils
from dataLoader import NERDataset
from model import ERNIE_BiLSTM_CRF
from run import loadDataset


import os
import config
import logging


def get_entities(seq):
    """
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        tag = chunk[0]
        type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'S':
        chunk_end = True
    # pred_label中可能出现这种情形
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_scores(y_true, y_pred):
    true_entities = set(get_entities(y_true))
    pred_entities = set(get_entities(y_pred))
    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    precision = nb_correct / nb_pred if nb_pred > 0 else 0
    recall = nb_correct / nb_true if nb_true > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    f_score = {}
    p_score = {}
    r_score = {}
    for label in config.labels:
        true_entities_label = set()
        pred_entities_label = set()
        for t in true_entities:
            if t[0] == label:
                true_entities_label.add(t)
        for p in pred_entities:
            if p[0] == label:
                pred_entities_label.add(p)
        nb_correct_label = len(true_entities_label & pred_entities_label)
        nb_pred_label = len(pred_entities_label)
        nb_true_label = len(true_entities_label)

        p_label = nb_correct_label / nb_pred_label if nb_pred_label > 0 else 0
        r_label = nb_correct_label / nb_true_label if nb_true_label > 0 else 0
        score_label = 2 * p_label * r_label / (p_label + r_label) if p_label + r_label > 0 else 0
        f_score[label] = score_label
        p_score[label] = p_label
        r_score[label] = r_label
    return f_score, f1_score, precision, recall, p_score, r_score


def bad_case(y_true, y_pred, data):
    if not os.path.exists(config.case_dir):
        os.system(r"touch {}".format(config.case_dir))  # 调用系统命令行来创建文件
    output = open(config.case_dir, 'w')
    for idx, (t, p) in enumerate(zip(y_true, y_pred)):
        if t == p:
            continue
        else:
            output.write("bad case " + str(idx) + ": \n")
            output.write("sentence: " + str(data[idx]) + "\n")
            output.write("golden label: " + str(t) + "\n")
            output.write("model pred: " + str(p) + "\n")
    logging.info("--------Bad Cases reserved !--------")



def evaluate(dev_loader, model, mode='dev'):
    # set model to evaluation mode
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(config.ernie_model_path, do_lower_case=True, skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            this_sent_data = [[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                               if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data]
            sent_data.extend(this_sent_data)

            tmp_sent_data = [[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices] for indices in batch_data]
            # 筛选出与label对齐的预标签表示（origin_sequence_output）
            origin_sequence_output = []
            for i, (tmp_data, starts) in enumerate(zip(tmp_sent_data, batch_token_starts)):
                # 使用 nonzero 和 squeeze 筛选出非零位置的输出
                valid_indices = starts.nonzero().squeeze(1)
                filtered_output = [tmp_data[idx] for idx in valid_indices.tolist()]
                origin_sequence_output.append(filtered_output)


            batch_masks = batch_data.gt(0)  # get padding mask, gt(x): get index greater than x
            label_masks = batch_tags.gt(-1)  # get padding mask, gt(x): get index greater than x
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]
            # (batch_size, max_len - padding_label_len)
            batch_output = model.crf.decode(batch_output, mask=label_masks)
            #修正
            chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]')
            fixed_batch_output = []
            for sent_row, output_row in zip(origin_sequence_output, batch_output):
                # 从第二列开始遍历
                tmp = output_row
                for col_idx in range(len(sent_row)):
                    value = sent_row[col_idx]
                    # 直接使用正则匹配，检查是否是非中文字符并更新
                    if not chinese_char_pattern.match(value) and output_row[col_idx] != 0:
                        tmp[col_idx] = 0
                fixed_batch_output.append(tmp)

            # (batch_size, max_len)
            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in fixed_batch_output])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])

    assert len(pred_tags) == len(true_tags)
    assert len(sent_data) == len(true_tags)
    # logging loss, f1 and report
    metrics = {}

    bad_case(true_tags, pred_tags, sent_data)
    f1_labels, f1, p, r, p_labels, r_labels = get_scores(true_tags, pred_tags)
    metrics['f1_labels'] = f1_labels
    metrics['p_labels'] = p_labels
    metrics['r_labels'] = r_labels
    metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    metrics['precision'] = p
    metrics['recall'] = r
    return metrics


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    utils.set_logger(config.log_dir)
    device = config.device
    logging.info("device: {}".format(device))
    word_test, label_test = loadDataset(config.test_dir)
    test_dataset = NERDataset(word_test, label_test, config)
    print('load data')
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = ERNIE_BiLSTM_CRF(config)
        weights = torch.load(config.model_dir)
        model.load_state_dict(weights)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    precision = val_metrics['precision']
    recall = val_metrics['recall']
    logging.info("test loss: {}, f1 score: {}, precision: {}, recall: {}".format(val_metrics['loss'], val_f1, precision, recall))
    val_f1_labels = val_metrics['f1_labels']
    val_p_labels = val_metrics['p_labels']
    val_r_labels = val_metrics['r_labels']

    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
        logging.info("precision score of {}: {}".format(label, val_p_labels[label]))
        logging.info("recall score of {}: {}".format(label, val_r_labels[label]))

test()