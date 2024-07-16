import sys

import torch
import torch.nn as nn
from torch import einsum
import copy
import os
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, precision_score
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
from prettytable import PrettyTable
from tqdm import tqdm


class Trainer(object):
    def __init__(self, model, optim, device, stage, train_dataloader, val_dataloader, test_dataloader, config):
        self.model = model
        self.optim = optim
        self.device = device
        if stage == 1:
            self.epochs = config.TRAIN.Stage1_MAX_EPOCH
        else:
            self.epochs = config.TRAIN.Stage23_MAX_EPOCH
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.step = 0
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.nb_training = len(self.train_dataloader)

        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0

        self.train_loss_epoch = []
        self.train_model_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config.TRAIN.OUTPUT_DIR
        self.stage = stage
        valid_metric_header = ["# Epoch", "AUROC", "AUPRC"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold"]

        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))

            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc = self.test(dataloader="val")

            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc]))
            self.val_table.add_row(val_lst)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Stage ' + str(self.stage) + ' Validation at Epoch ' + str(self.current_epoch), "with AUROC "+ str(auroc) + " AUPRC " + str(auprc))
        auroc, auprc, f1, sensitivity, specificity, accuracy, thred_optim, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, thred_optim]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + " with AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + str(thred_optim))
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = thred_optim
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()

        return self.test_metrics, self.best_epoch

    def save_result(self):
        if self.config.TRAIN.SAVE_MODEL:
            torch.save(self.best_model.state_dict(), os.path.join(self.output_dir, f"stage_{self.stage}_best_epoch_{self.best_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, 'Stage_' + str(self.stage) +"_valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, 'Stage_' + str(self.stage) +"_test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, 'Stage_' + str(self.stage) +"_train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        loop = tqdm(self.train_dataloader, colour='#ff4777', file=sys.stdout, ncols=120)
        loop.set_description(f'Stage {self.stage} Train Epoch[{self.current_epoch}/{self.epochs}]')
        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1
            input_drugs = batch['batch_inputs_drug']
            input_proteins = batch['batch_inputs_pr']
            labels = torch.tensor(batch['labels'])

            drug_ids = input_drugs['input_ids']
            drug_padding_mask = input_drugs['attention_mask']
            pr_ids = input_proteins['input_ids']
            pr_padding_mask = input_proteins['attention_mask']

            drug_ids, drug_padding_mask, pr_ids, pr_padding_mask, labels = \
                (drug_ids.to(self.device), drug_padding_mask.to(self.device), pr_ids.to(self.device),
                 pr_padding_mask.to(self.device), labels.to(self.device))
            output = self.model(drug_ids, drug_padding_mask, pr_ids, pr_padding_mask)
            n, binary_loss = binary_cross_entropy(output['logits'], labels.float())

            if self.stage == 1:
                masked_drug_ids = input_drugs['masked_input_ids'].to(self.device)
                drug_labels = input_drugs['masked_drug_labels'].to(self.device)
                output = self.model(masked_drug_ids, drug_padding_mask, pr_ids, pr_padding_mask, mlm=True)
                mask_loss = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_logits'], drug_labels)
                loss = binary_loss + 0.1 * mask_loss
            else:
                loss = binary_loss
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Stage ' + str(self.stage) + ' Training at Epoch ' + str(self.current_epoch) + ' with average loss ' + str(loss_epoch))
        return loss_epoch

    def train_cross_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        loop = tqdm(self.train_dataloader, colour='#ff4777', file=sys.stdout, ncols=120)
        loop.set_description(f'Stage {self.stage} Train Epoch[{self.current_epoch}/{self.epochs}]')
        for step, batch in enumerate(loop):
            self.optim.zero_grad()
            self.step += 1
            input_drugs = batch['batch_inputs_drug']
            input_proteins = batch['batch_inputs_pr']
            labels = torch.tensor(batch['labels'])

            drug_ids = input_drugs['input_ids']
            drug_padding_mask = input_drugs['attention_mask']
            pr_ids = input_proteins['input_ids']
            pr_padding_mask = input_proteins['attention_mask']

            drug_ids, drug_padding_mask, pr_ids, pr_padding_mask, labels = \
                (drug_ids.to(self.device), drug_padding_mask.to(self.device), pr_ids.to(self.device),
                 pr_padding_mask.to(self.device), labels.to(self.device))
            output = self.model(drug_ids, drug_padding_mask, pr_ids, pr_padding_mask)
            n, binary_loss = binary_cross_entropy(output['logits'], labels.float())

            if self.stage == 1:
                masked_drug_ids = input_drugs['masked_input_ids'].to(self.device)
                drug_labels = input_drugs['masked_drug_labels'].to(self.device)
                output = self.model(masked_drug_ids, drug_padding_mask, pr_ids, pr_padding_mask, mlm=True)
                mask_loss = nn.CrossEntropyLoss(ignore_index=-1)(output['drug_logits'], drug_labels)
                loss = binary_loss + 0.1 * mask_loss
            else:
                loss = binary_loss
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()

        loss_epoch = loss_epoch / num_batches
        print('Stage ' + str(self.stage) + ' Training at Epoch ' + str(self.current_epoch) + ' with average loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        y_label, y_pred = [], []
        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")
        loop = tqdm(data_loader, colour='#f47983', file=sys.stdout)
        with torch.no_grad():
            self.model.eval()
            if dataloader == "val":
                loop.set_description(f'Stage {self.stage} Validation')
            elif dataloader == "test":
                loop.set_description(f'Stage {self.stage} Test')
            for step, batch in enumerate(loop):
                input_drugs = batch['batch_inputs_drug']
                input_proteins = batch['batch_inputs_pr']
                labels = batch['labels']

                drug_ids = input_drugs['input_ids']
                drug_padding_mask = input_drugs['attention_mask']
                pr_ids = input_proteins['input_ids']
                pr_padding_mask = input_proteins['attention_mask']

                drug_ids, drug_padding_mask, pr_ids, pr_padding_mask = \
                drug_ids.to(self.device), drug_padding_mask.to(self.device), \
                pr_ids.to(self.device), pr_padding_mask.to(self.device)

                if dataloader == "val":
                    logits = self.model(drug_ids, drug_padding_mask, pr_ids, pr_padding_mask)['logits']
                elif dataloader == "test":
                    logits = self.best_model(drug_ids, drug_padding_mask, pr_ids, pr_padding_mask)['logits']
                n, loss = binary_cross_entropy(logits, torch.tensor(labels).to(self.device).float())
                y_label = y_label + labels
                y_pred = y_pred + n.tolist()
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)
        if dataloader == "test":
            fpr, tpr, thresholds = roc_curve(y_label, y_pred)
            prec, recall, _ = precision_recall_curve(y_label, y_pred)
            precision = tpr / (tpr + fpr)
            f1 = 2 * precision * tpr / (tpr + precision + 0.000001)
            thred_optim = thresholds[5:][np.argmax(f1[5:])]
            y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
            cm1 = confusion_matrix(y_label, y_pred_s)
            accuracy = (cm1[0, 0] + cm1[1, 1]) / sum(sum(cm1))
            sensitivity = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
            specificity = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])

            precision1 = precision_score(y_label, y_pred_s)
            return auroc, auprc, np.max(f1[5:]), sensitivity, specificity, accuracy, thred_optim, precision1
        else:
            return auroc, auprc
def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    return loss
# def get_c_loss(drug_fs, pr_fs, weight, tem, labels):
#     alpha = 0.2
#     temp = 0.06
#     temp2 = 1000
#     drug_fs = F.normalize(drug_fs, dim=1)
#     pr_fs = F.normalize(pr_fs, dim=1)
#     mask = labels.bool()
#     drug_pos = drug_fs[mask]
#     drug_neg = drug_fs[~mask]
#     drug_fs = torch.cat((drug_pos, drug_neg), dim=0)
#     pr_pos = pr_fs[mask]
#     pr_neg = pr_fs[~mask]
#     pr_fs = torch.cat((pr_pos, pr_neg), dim=0)
#     c_drug_logits = (drug_pos @ pr_fs.T) / temp
#     c_pr_logits = (pr_pos @ drug_fs.T) / temp
#     images_similarity = pr_pos @ pr_fs.T
#     texts_similarity = drug_pos @ drug_fs.T
#     new_tensor = torch.eye(drug_pos.size(0)).to(labels.device)
#     x = torch.zeros(drug_pos.size(0), drug_neg.size(0)).to(labels.device)
#     new_tensor = torch.cat((new_tensor, x), dim=1)
#     targets_text = F.softmax(texts_similarity * temp, dim=-1)*alpha + (1-alpha) * new_tensor
#     targets_img = F.softmax(images_similarity * temp, dim=-1)*alpha + (1-alpha) * new_tensor
#     texts_loss = cross_entropy(c_drug_logits, targets_text, reduction='none')
#     images_loss = cross_entropy(c_pr_logits, targets_img, reduction='none')
#     c_loss = (images_loss.mean() + texts_loss.mean()) / 2.0  # shape: (batch_size)

    # drug_fs = torch.cat((drug_neg, drug_pos), dim=0)
    # pr_fs = torch.cat((pr_neg, pr_pos), dim=0)
    # c_drug_logits = (drug_neg @ pr_fs.T) / temp2
    # c_pr_logits = (pr_neg @ drug_fs.T) / temp2
    # images_similarity2 = pr_neg @ pr_fs.T
    # texts_similarity2 = drug_neg @ drug_fs.T
    # targets_text2 = F.softmax(texts_similarity2 * temp2, dim=-1)
    # targets_img2 = F.softmax(images_similarity2 * temp2, dim=-1)
    # texts_loss2 = cross_entropy(c_drug_logits, targets_text2, reduction='none')
    # images_loss2 = cross_entropy(c_pr_logits, targets_img2, reduction='none')
    # c_loss2 = (images_loss2.mean() + texts_loss2.mean()) / 2.0  # shape: (batch_size)

    # c_loss = c_loss2 + c_loss
    # return c_loss

# def get_c_loss(drug_fs, pr_fs, weight, tem, labels):
#     lam = 0.5
#     q = 0.1
#     temp = 1.0
#     # normalize
#     drug_fs = nn.functional.normalize(drug_fs, dim=1)
#     pr_fs = nn.functional.normalize(pr_fs, dim=1)
#
#     mask = labels.bool()
#
#     neg = torch.exp(drug_fs @ pr_fs.T / temp)
#     pos = torch.exp(torch.sum(drug_fs[mask] * pr_fs[mask], dim=-1)/ temp)
#
#     pos = -(pos ** q) / q
#     neg = ((lam * (neg.sum(1))) ** q) / q
#     loss = pos.mean() + neg.mean()
#
#     return loss * (2 * temp)

# def get_c_loss(drug_fs, pr_fs, weight, tem, labels):
#     c_logits = (drug_fs @ pr_fs.T) / tem
#     label = label_bin(labels).flatten()
#     pred = c_logits.flatten().unsqueeze(1)
#
#     n, loss = binary_cross_entropy(pred, label)
#     return loss

# def label_bin(labels):
#     BZ = labels.size(0)
#     new_tensor = torch.ones(BZ, BZ).to(labels.device)
#
#     # 创建一个对角线索引掩码，形状为[BZ, BZ]，对角线为True，其余为False
#     diagonal_mask = torch.eye(BZ, dtype=torch.bool).to(labels.device)
#
#     update_mask = labels.bool().unsqueeze(1) & diagonal_mask
#     new_tensor[update_mask.to(labels.device)] = 0
#
#     return (1-new_tensor).abs()
# def get_c_loss(drug_fs, pr_fs, weight, tem, labels):
#     tem = 1.0
#     alpha = 0.6
#     drug_fs = F.normalize(drug_fs, dim=1)
#     pr_fs = F.normalize(pr_fs, dim=1)
#     texts_similarity = drug_fs @ drug_fs.T
#     images_similarity = pr_fs @ pr_fs.T
#     new_tensor = torch.eye(labels.size(0)).to(labels.device)
#     targets = alpha * F.softmax((texts_similarity + images_similarity)/2 * tem, dim=-1) + (1 - alpha) * new_tensor
#     mat = adjust_target(labels, -1)
#     c_logits = (drug_fs @ pr_fs.T) / tem * mat
#     texts_loss = cross_entropy(c_logits, targets, reduction='none').mean()
#     images_loss = cross_entropy(c_logits.T, targets.T, reduction='none').mean()
#     c_loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
#     return c_loss
#
# def adjust_target(labels,value):
#     BZ = labels.size(0)
#     new_tensor = torch.ones(BZ, BZ).to(labels.device)
#
#     # 创建一个对角线索引掩码，形状为[BZ, BZ]，对角线为True，其余为False
#     diagonal_mask = torch.eye(BZ, dtype=torch.bool).to(labels.device)
#
#     # 根据标签值为0的位置设置对角线上的相应位置为0.1
#     # 这里利用逻辑非~来找到标签为0的位置，然后与对角线掩码进行逻辑与操作
#     update_mask = ~labels.bool().unsqueeze(1) & diagonal_mask
#     new_tensor[update_mask.to(labels.device)] = value
#     # update_mask = ~labels.bool().unsqueeze(1) & diagonal_mask
#     # new_tensor[,:]=2
#     # new_tensor[:,]=2
#
#
#     return new_tensor