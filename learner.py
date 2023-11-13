from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tensorboardX
import os
import torch.optim as optim
import sys
from sklearn.metrics import roc_curve, roc_auc_score
import models
import utils
# from data.util import get_dataset, IdxDataset
# from module.loss import GeneralizedCELoss
# from module.util import get_model
# from util import EMA
from datasets import get_dataset
import datetime


class learner():
    def __init__(self, args):
        self.args = args
        self.train_df, self.test_df, self.white_test_df, self.other_test_df = utils.get_dataframe(args)
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.device = "cuda:{}".format(args.device) if args.device != -1 else 'cpu'
        self.train_loader, self.test_loader, self.w_loader, self.o_loader = None, None, None, None
        self.disease = args.disease
        self.w_a, self.o_a, self.all_a, self.e_a = None, None, None, None
        save_dir, save_name = None, None
        self.train_set = None
        if args.model == 1:
            save_dir = './saved_result/vanilla/'
            save_name = '{}.txt'.format(self.args.exp_name)
        elif args.model == 2:
            save_dir = './saved_result/debias/'
            save_name = '{}.txt'.format(self.args.exp_name)
        else:
            save_dir = './saved_result/race/'
            save_name = '{}.txt'.format(self.args.exp_name)
        self.result_save_path = os.path.join(save_dir, save_name)
        utils.make_dir(save_dir)

        # Dataloader
        if args.model == 1:
            train_dataset = get_dataset('CT', self.train_df, self.data_path, 'train')
            val_dataset = get_dataset('CT', self.test_df, self.data_path, 'val')
            w_dataset = get_dataset('CT', self.white_test_df, self.data_path, 'test')
            o_dataset = get_dataset('CT', self.other_test_df, self.data_path, 'test')



        else:

            train_dataset = get_dataset('CT2', self.train_df, self.data_path, 'train')
            val_dataset = get_dataset('CT2', self.test_df, self.data_path, 'val')
            self.train_set = train_dataset
            w_dataset = get_dataset('CT2', self.white_test_df, self.data_path, 'test')
            o_dataset = get_dataset('CT2', self.other_test_df, self.data_path, 'test')

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                                        num_workers=self.num_workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                                                      num_workers=self.num_workers, pin_memory=True)
        self.w_loader = torch.utils.data.DataLoader(w_dataset, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers, pin_memory=True)
        self.o_loader = torch.utils.data.DataLoader(o_dataset, batch_size=self.batch_size, shuffle=True,
                                                    num_workers=self.num_workers, pin_memory=True)
        # TensorBoardX
        if args.tensorboard == 1:
            self.writer = tensorboardX.SummaryWriter(f'result/summary/{args.exp_name}/{args.disease}')

    def write_result(self, string):
        with open(self.result_save_path, 'a+') as file:
            sys.stdout = file
            print(string)
        sys.stdout = sys.__stdout__

    def test_basic(self, path_name, key):
        device = self.device
        model = models.ResNet34(num_classes=2).to(device)
        model.load_state_dict(torch.load(path_name))
        model.eval()

        test_loader = None
        if key == 'all':
            test_loader = self.val_loader
        elif key == 'w':
            test_loader = self.w_loader
        elif key == 'o':
            test_loader = self.o_loader

        ground_truth = []
        predict_score = []
        test_num = torch.zeros(1).to(device)
        ac = torch.zeros(1).to(device)
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                _images, _labels = test_data
                outputs = model(_images.to(device))
                scores = outputs[:, 1]
                predict_y = torch.max(outputs, dim=1)[1]
                # print('score={},y={}'.format(scores, predict_y))
                ground_truth += _labels.cpu().numpy().tolist()
                predict_score += scores.cpu().numpy().tolist()
                # print(len(ground_truth),len(predict_score))# 这里有问题
                test_num += len(predict_y)
                ac += torch.eq(predict_y, _labels.to(device)).sum().item()

        test_accurate = ac / test_num
        roc_auc = roc_auc_score(ground_truth, predict_score)
        fpr, tpr, thresholds = roc_curve(ground_truth, predict_score)

        if key == 'all':
            self.all_a = test_accurate
        elif key == 'w':
            self.w_a = test_accurate
        elif key == 'o':
            self.o_a = test_accurate

        with open(self.result_save_path, 'a+') as file:
            sys.stdout = file
            key1 = key
            if key == 'all':
                key1 = 'l'
            print('[ {} ]'.format(key1), end="")
            print('[ accurate , roc_auc ]= [ {:6f} , {:6f} ]'.format(float(test_accurate), roc_auc))
            if key == 'all':
                print("[ {} ][ accurate , roc_auc ]= [ {:6f} , {:6f} ]".format('e', float(self.e_a), float(0.000000)))
        sys.stdout = sys.__stdout__


    def train_basic(self):
        steps = 0
        device = self.device
        epochs = self.args.epochs
        lr = self.args.lr

        best_acc = torch.zeros(1).to(device)

        save_dir = r'./checkpoints/vanilla/'
        utils.make_dir(save_dir)
        save_name = 'Best' + '_' + str(self.disease) + '_' + 'model1'
        save_path = save_dir + save_name + r'.pth'

        model = models.ResNet34(num_classes=2).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        train_steps = len(self.train_loader)
        for epoch in range(epochs):
            # train
            model.train()
            running_loss = torch.zeros(1).to(device)
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data

                logits = model(images.to(device))
                #
                loss = loss_function(logits, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if self.args.tensorboard == 1:
                    self.writer.add_scalar(f"loss/steps", loss.item(), steps)
                    steps += 1
                train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            model.eval()
            acc = torch.zeros(1).to(device)
            val_num = torch.zeros(1).to(device)
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                    val_num += len(predict_y)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num

            print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate

                torch.save(model.state_dict(), save_path)

            if self.args.tensorboard == 1:
                self.writer.add_scalar(f"acc/epoch", val_accurate, epoch)
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M")
        self.write_result("Time [ {} ]  Disease [  {}  ]  ".format(date_string, self.disease))
        self.test_basic(save_path, 'w')
        self.test_basic(save_path, 'o')

        self.e_a = (self.o_a + self.w_a) / 2.0
        self.test_basic(save_path, 'all')
        self.write_result('-\n')

    def test_debias(self, path_name, key):
        device = self.device
        model = models.ResNet34(num_classes=2).to(device)
        model.load_state_dict(torch.load(path_name))
        model.eval()

        test_loader = None
        if key == 'all':
            test_loader = self.val_loader
        elif key == 'w':
            test_loader = self.w_loader
        elif key == 'o':
            test_loader = self.o_loader

        ground_truth = []
        predict_score = []
        test_num = torch.zeros(1).to(device)
        ac = torch.zeros(1).to(device)
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                index, _images, _labels, _races = test_data
                _images = _images.to(device)
                f_b = model.extract(_images)
                pred_d = model.d_fc(f_b)
                predict_y = torch.max(pred_d, dim=1)[1]
                scores = pred_d[:, 1]

                # print('score={},y={}'.format(scores, predict_y))
                ground_truth += _labels.cpu().numpy().tolist()
                predict_score += scores.cpu().numpy().tolist()
                # print(len(ground_truth),len(predict_score))# 这里有问题
                test_num += len(predict_y)
                ac += torch.eq(predict_y, _labels.to(device)).sum().item()

        test_accurate = ac / test_num
        roc_auc = roc_auc_score(ground_truth, predict_score)
        fpr, tpr, thresholds = roc_curve(ground_truth, predict_score)

        if key == 'all':
            self.all_a = test_accurate
        elif key == 'w':
            self.w_a = test_accurate
        elif key == 'o':
            self.o_a = test_accurate

        with open(self.result_save_path, 'a+') as file:
            sys.stdout = file
            key1 = key
            if key == 'all':
                key1 = 'l'
            print('[ {} ]'.format(key1), end="")
            print('[ accurate , roc_auc ]= [ {:6f} , {:6f} ]'.format(float(test_accurate), roc_auc))
            if key == 'all':
                print("[ {} ][ accurate , roc_auc ]= [ {:6f} , {:6f} ]".format('e', float(self.e_a), float(0.000000)))
        sys.stdout = sys.__stdout__

    def train_debias(self):

        steps = 0
        device = self.device
        epochs = self.args.epochs
        lr = self.args.lr
        weight_decay = self.args.weight_decay
        best_acc = torch.zeros(1).to(device)

        save_dir = r'./checkpoints/debias/'
        utils.make_dir(save_dir)
        save_name_1 = str(self.args.exp_name) + 'Best' + '_' + str(self.disease) + '_' + 'model2-1'
        save_name_2 = str(self.args.exp_name) + 'Best' + '_' + str(self.disease) + '_' + 'model2-2'
        save_path_1 = save_dir + save_name_1 + r'.pth'
        save_path_2 = save_dir + save_name_2 + r'.pth'

        # sample loss ema
        # print(self.train_set[-1])
        sl_b = utils.EMA(torch.LongTensor(self.train_set.races[:]), num_classes=2, alpha=self.args.ema_alpha)
        sl_i = utils.EMA(torch.LongTensor(self.train_set.races[:]), num_classes=2, alpha=self.args.ema_alpha)

        model_b = models.ResNet34(num_classes=2).to(device)
        model_i = models.ResNet34(num_classes=2).to(device)

        CE = nn.CrossEntropyLoss(reduction='none')
        GCE = utils.GeneralizedCELoss()

        optimizer_b = optim.Adam(model_b.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_i = optim.Adam(model_i.parameters(), lr=lr, weight_decay=weight_decay)

        scheduler_b = optim.lr_scheduler.MultiStepLR(optimizer_b, milestones=self.args.mile_d, gamma=0.1)
        scheduler_i = optim.lr_scheduler.MultiStepLR(optimizer_i, milestones=self.args.mile_r, gamma=0.1)

        train_steps = len(self.train_loader)

        for epoch in range(epochs):
            model_b.train()
            model_i.train()
            running_loss = torch.zeros(1).to(device)
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, datas in enumerate(train_bar):
                index, data, label, race = datas
                index = torch.Tensor(index).long().to(device)
                data = data.to(device)
                label = label.to(device)
                race = race.to(device)

                f_i = model_i.extract(data)
                f_b = model_b.extract(data)

                f_align = torch.cat((f_b.detach(), f_i), dim=1)
                f_conflict = torch.cat((f_b, f_i.detach()), dim=1)

                pred_a = model_i.r_fc(f_align)
                pred_c = model_b.r_fc(f_conflict)

                # lda : loss_dis_align
                # ldc : loss_dis_conflict
                # ldd : loss_dis_disease

                lda = CE(pred_a, race).detach()
                ldc = CE(pred_c, race).detach()

                pred_d = model_b.d_fc(f_b)
                pred_r = model_i.r2_fc(f_i)
                lambda_d = 0
                lambda_r = 0.0
                if epoch > self.args.swap_epoch:
                    lambda_d = self.args.lambda_d
                    lambda_r = 0

                ldd = CE(pred_d, label)

                ldr = CE(pred_r, race)

                # class-wise normalize
                sl_i.update(lda, index)
                sl_b.update(ldc, index)
                lda = sl_i.parameter[index].clone().detach()
                ldc = sl_b.parameter[index].clone().detach()
                lda = lda.to(device)
                ldc = ldc.to(device)
                for c in range(2):
                    class_index = torch.where(race == c)[0].to(self.device)
                    max_loss_align = sl_i.max_loss(c)
                    max_loss_conflict = sl_b.max_loss(c)
                    lda[class_index] /= max_loss_align
                    ldc[class_index] /= max_loss_conflict

                loss_weight = ldc / (ldc + lda + 1e-8)
                lda = CE(pred_a, race) * loss_weight.to(device)
                ldc = GCE(pred_c, race)

                loss_dis = lda.mean() + ldc.mean()
                loss = loss_dis + ldd.mean() * lambda_d + ldr.mean() * lambda_r

                # lsa : loss_swap_align
                # lsc : loss_swap_conflict
                lsa = torch.tensor([0]).float()
                lsc = torch.tensor([0]).float()

                lambda_swap = 0
                if epoch > self.args.swap_epoch:
                    indices = np.random.permutation(f_b.size(0))
                    fb_swap = f_b[indices]
                    race_swap = race[indices]

                    f_mix_align = torch.cat((fb_swap.detach(), f_i), dim=1)
                    f_mix_conflict = torch.cat((fb_swap, f_i.detach()), dim=1)
                    # pred_mix_align
                    pma = model_i.r_fc(f_mix_align)
                    pmc = model_b.r_fc(f_mix_conflict)

                    lsa = CE(pma, race) * loss_weight.to(device)
                    lsc = GCE(pmc, race_swap)
                    lambda_swap = self.args.lambda_swap

                loss_swap = lsa.mean() + lsc.mean()
                loss = loss + lambda_swap * loss_swap

                optimizer_i.zero_grad()
                optimizer_b.zero_grad()
                loss.backward()
                optimizer_i.step()
                optimizer_b.step()
                scheduler_b.step()
                scheduler_i.step()

                running_loss += loss.item()
                if self.args.tensorboard == 1:
                    self.writer.add_scalar(f"loss/steps", loss.item(), steps)
                    self.writer.add_scalar(f"lda/steps", lda.mean(), steps)
                    self.writer.add_scalar(f"ldc/steps", ldc.mean(), steps)
                    self.writer.add_scalar(f"lsa/steps", lsa.mean(), steps)
                    self.writer.add_scalar(f"lsc/steps", lsc.mean(), steps)
                    steps += 1

                train_bar.desc = "epoch[{}/{}] loss:{:.4f} ".format(epoch + 1, epochs, loss)

            model_b.eval()
            model_i.eval()

            acc = torch.zeros(1).to(device)
            acc2 = torch.zeros(1).to(device)
            val_num = torch.zeros(1).to(device)
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    index, val_images, val_labels, val_races = val_data
                    val_images = val_images.to(device)
                    f_b = model_b.extract(val_images)
                    pred_d = model_b.d_fc(f_b)
                    predict_y = torch.max(pred_d, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    f_i = model_i.extract(val_images)
                    pred_r = model_i.r2_fc(f_i)
                    predict_r = torch.max(pred_r, dim=1)[1]
                    acc2 += torch.eq(predict_r, val_races.to(device)).sum().item()

                    val_num += len(predict_y)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            race_acc = acc2 / val_num
            print('[epoch %d] train_loss: %.5f  val_acc: %.5f val_acc2: %.5f' %
                  (epoch + 1, running_loss / train_steps, val_accurate, race_acc))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model_i.state_dict(), save_path_1)
                torch.save(model_b.state_dict(), save_path_2)

            if self.args.tensorboard == 1:
                self.writer.add_scalar(f"acc/epoch", val_accurate, epoch)

        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M")
        self.write_result("Time [ {} ]  Disease [  {}  ]  ".format(date_string, self.disease))
        self.test_debias(save_path_2, 'w')
        self.test_debias(save_path_2, 'o')
        self.e_a = (self.w_a + self.o_a) / 2.0
        self.test_debias(save_path_2, 'all')
        self.write_result('-\n')

# race claasification
    def test_race(self, path_name):
        device = self.device
        model = models.ResNet34(num_classes=3).to(device)
        model.load_state_dict(torch.load(path_name))
        model.eval()

        test_loader = self.val_loader

        ground_truth = []
        predict_score = []
        test_num = torch.zeros(1).to(device)
        ac = torch.zeros(1).to(device)
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                _index, _images, _labels, _races = test_data
                outputs = model(_images.to(device))
                scores = outputs[:]
                predict_y = torch.max(outputs, dim=1)[1]
                # print('score={},y={}'.format(scores, predict_y))
                ground_truth += _races.cpu().numpy().tolist()
                predict_score += scores.cpu().numpy().tolist()
                # print(len(ground_truth),len(predict_score))# 这里有问题
                test_num += len(predict_y)
                ac += torch.eq(predict_y, _races.to(device)).sum().item()

        test_accurate = ac / test_num
        roc_auc = roc_auc_score(ground_truth, predict_score, multi_class='ovr')
        # fpr, tpr, thresholds = roc_curve(ground_truth, predict_score, multi_class='ovr')

        with open(self.result_save_path, 'a+') as file:
            sys.stdout = file
            print('[ accurate , roc_auc ]= [ {:6f} , {:6f} ]'.format(float(test_accurate), roc_auc))
        sys.stdout = sys.__stdout__

    def train_race(self):
        steps = 0
        device = self.device
        epochs = self.args.epochs
        lr = self.args.lr

        best_acc = torch.zeros(1).to(device)

        save_dir = r'./checkpoints/races/'
        utils.make_dir(save_dir)
        save_name = 'Best' + '_' + str(self.disease) + '_' + 'model1'
        save_path = save_dir + save_name + r'.pth'

        model = models.ResNet34(num_classes=3).to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=model.parameters(), lr=lr)

        train_steps = len(self.train_loader)
        for epoch in range(epochs):
            # train
            model.train()
            running_loss = torch.zeros(1).to(device)
            train_bar = tqdm(self.train_loader, file=sys.stdout)
            for step, datas in enumerate(train_bar):
                index, data, label, race = datas

                logits = model(data.to(device))
                #
                loss = loss_function(logits, race.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if self.args.tensorboard == 1:
                    self.writer.add_scalar(f"loss/steps", loss.item(), steps)
                    steps += 1
                train_bar.desc = "train epoch[{}/{}] loss:{:.5f}".format(epoch + 1,
                                                                         epochs,
                                                                         loss)
            model.eval()
            acc = torch.zeros(1).to(device)
            val_num = torch.zeros(1).to(device)
            with torch.no_grad():
                val_bar = tqdm(self.val_loader, file=sys.stdout)
                for val_data in val_bar:
                    index_v, val_images, val_labels, val_races = val_data
                    outputs = model(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_races.to(device)).sum().item()
                    val_num += len(predict_y)
                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num

            print('[epoch %d] train_loss: %.5f  val_accuracy: %.5f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate

                torch.save(model.state_dict(), save_path)

            if self.args.tensorboard == 1:
                self.writer.add_scalar(f"acc/epoch", val_accurate, epoch)
        now = datetime.datetime.now()
        date_string = now.strftime("%Y-%m-%d %H:%M")
        self.write_result("Time [ {} ]  Disease [  {}  ]  ".format(date_string, self.disease))
        self.test_race(save_path)
        self.write_result('-\n')
