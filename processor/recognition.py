#!/usr/bin/env python
# pylint: disable=W0201
from pyexpat import features
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class
import torch.nn.functional as F

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss_1 = nn.CrossEntropyLoss()
        self.loss_2 = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        #kld loss 선언
        
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
            self.optimizer_2 = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)    
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self, chunk=1):
        
        
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []

        for data, label in loader:
            # get data
            # datas = data.chunk(chunk, dim=2)
            #print(datas[0].shape)
            # label = label.long().to(self.dev)
            # data = d[:, :, 100:300, :]
            # data = data.float().to(self.dev)
            
            # for d in datas:
                
                # d = d[:,:,100:300,:]
                # d = d.float().to(self.dev)
            
                # if self.kd_model != "null":
                #     self.kd_model.eval()
                    
                #     #for kd learning
                #     with torch.no_grad():
                #         _, teacher_logits = self.kd_model(d)
                    
                #     output, gcn_feature = self.model(d)
                    
                    
                #     fc_layer  = nn.Linear(32, 7)
                #     fc_layer.cuda()
                    
                #     teacher_logits = torch.sigmoid(teacher_logits)
                #     teacher_logits = fc_layer(teacher_logits)
                    
                #     gcn_feature = torch.sigmoid(gcn_feature)
                #     gcn_feature = fc_layer(gcn_feature)

                    
                #     #Soften the student logits by applying softmax first and log() second
                #     soft_targets = nn.functional.softmax(teacher_logits / 2, dim=-1)
                #     soft_prob = nn.functional.log_softmax(gcn_feature / 2, dim=-1)
                    
                #     soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (2**2)
                #     #hidden_rep_loss = self.cosine_loss(gcn_feature, teacher_logits, target=torch.ones(d.size(0)).to((self.dev)))

                #     label_loss = self.loss_1(output, label)
                    
                    
                #     loss = 0.25 * soft_targets_loss + 0.75 * label_loss
                #     #loss = 0.25 * hidden_rep_loss + 0.75 * label_loss
                    
                #                                 # backward
                #     self.optimizer.zero_grad()
                #     loss.backward()
                #     self.optimizer.step()

                #     # statistics
                #     self.iter_info['loss'] = loss.data.item()
                #     self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                #     loss_value.append(self.iter_info['loss'])
                #     self.show_iter_info()
                #     self.meta_info['iter'] += 1
                    
                #     #kd learning 끝
                #     #___________________________________
                    
                    
                # else:
                    # d = d.float().to(self.dev)
                    
                    # output, gcn_feature = self.model(d)
                    
                    # #print( "d, label ..................")
                    # #print(output.shape)
                    # #print(label)
                    # loss = self.loss_1(output, label)
                    # # inference
                    #             # backward
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()

                    # # statistics
                    # self.iter_info['loss'] = loss.data.item()
                    # self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                    # loss_value.append(self.iter_info['loss'])
                    # self.show_iter_info()
                    # self.meta_info['iter'] += 1
                
            self.model.train()
            data = data[:,:,120:320,:]
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            # forward
            
            #앞 뒤 데이터 자르기.
            
            output, gcn_feature = self.model(data)
            
            loss = self.loss_1(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
        
    def test(self, evaluation=True, chunk=1):
        correct=0
        total_data = 0
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        # result_frag = []
        label_frag = []

        for data, label in loader:
            # get data

            data = data[:,:,250:300,:]

            data = data.float().to(self.dev)
            label = label.long().to(self.dev)
            # print("test label: ", label)

            # inference
            with torch.no_grad():
                output,_ = self.model(data)
                _, output_index = torch.max(output, 1)  
                # print("prediction: ", output_index)
               
                if evaluation:
                    for i in range(len(output_index)):
                        if (output_index[i] == label[i]):
                            correct +=1
                    total_data += len(output_index)
            
                    
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        
        return correct, (100*(correct/total_data))
    def kd_train(self, chunk=1):
        
        
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['first train']
        skloader = self.data_loader['second train']
        loss_value = []
        for first, second in zip(loader, skloader) :
            # get data
            # datas = data.chunk(chunk, dim=2)
            #print(datas[0].shape)
            label = first[1]
            data = first[0]
            skdata = second[0]
            sklabel = second[1]
            
            
            
            label = label.long().to(self.dev)
            datas= data[:, :, 175:225, :]
            
            skdata = skdata[:, :, 120:320, :]
            skdata = skdata.float().to(self.dev)
            sklabel = sklabel.long().to(self.dev)
            # print("ssssssssssssssssss")
            print(label, sklabel)

            datat= data[:, :, 120:320, :]
            data = data.float().to(self.dev)
            datas = datas.float().to(self.dev)
            datat = skdata
            # datat = datat.float().to(self.dev)
            # d = d[:,:,100:300,:]
            # d = d.float().to(self.dev)
        
        
            # if self.kd_model != "null":
            self.kd_model.eval()
            
            #for kd learning
            with torch.no_grad():
                teacher_logits, teacher_features = self.kd_model(datat)
            
            output, gcn_feature = self.model(datas)
                
                
            
            fc_layer  = nn.Conv2d(32, 7, 1)
            fc_layer.cuda()

            #output = output.squeeze(2).squeeze(2)
            #output = self.softmax(output)
        
            # teacher_logits = torch.sigmoid(teacher_logits)
            teacher_features_fc = fc_layer(teacher_features)
            teacher_features_fc = teacher_features_fc.squeeze(2).squeeze(2)
            teacher_features = teacher_features.squeeze(2).squeeze(2)

            
            # gcn_feature = torch.sigmoid(gcn_feature)
            gcn_feature_fc = fc_layer(gcn_feature)
            gcn_feature_fc = gcn_feature_fc.squeeze(2).squeeze(2)
            gcn_feature = gcn_feature.squeeze(2).squeeze(2)
                
            #Soften the student logits by applying softmax first and log() second
            # teacher_probs  = nn.functional.softmax(teacher_features_fc / 2, dim=-1)
            # student_probs  = nn.functional.log_softmax(gcn_feature_fc / 2, dim=-1)

            # f = F.kl_div()(input=torch.log(student_probs),
            #         target=teacher_probs, 
            #         reduction='batchmean')  
              
             #soft_targets_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (2**2)
            hidden_rep_loss = self.cosine_loss(gcn_feature, teacher_features, target=torch.ones(gcn_feature.size(0)).to((self.dev)))
            mse_hidden_rep_loss =  self.loss_2(gcn_feature, teacher_features)
            soft_loss = self.kl_loss(F.log_softmax(gcn_feature_fc/2, dim=-1), F.softmax(teacher_features_fc/2, dim=-1))
            cosine_rep_loss = self.cosine_loss(gcn_feature_fc, teacher_features_fc, target=torch.ones(gcn_feature_fc.size(0)).to((self.dev)))
            
            label_loss = self.loss_1(output, label)
            # print("soft loss : ", soft_loss)
            # print("hidden_rep_loss : ",hidden_rep_loss)
            
            # loss = 0.25 * f + 0.75 * label_loss
            # loss = 0.05 * hidden_rep_loss + 0.05 * f + 0.9 * label_loss
            loss = 0.05*hidden_rep_loss + 0.05*soft_loss + 0.9*label_loss
            # loss = 0.1*cosine_rep_loss + 0.9*label_loss
            # loss = 0.05*hidden_rep_loss + 0.05*cosine_rep_loss + 0.9*label_loss

            # loss = 0.2 * soft_loss + 0.8*label_loss
            # loss = 0.1 * hidden_rep_loss + 0.9 * label_loss
            # loss = 0.25 * hidden_rep_loss + f
            # loss = 0.25 * mse_hidden_rep_loss + 0.75 * label_loss
                                        # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            
            #kd learning 끝
            #___________________________________
                



        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
        
    def slidingtest(self, evaluation=True, chunk=1):
        self.relu = torch.nn.ReLU()
        correct=0
        total_data = 0
        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        # result_frag = []
        label_frag = []

        for data, label in loader:
            # get data
            
            datas = []
            for i in range(0,350+1, 10 ):
                datas.append(data[:, :, i:i+50, :])

            label = label.long().to(self.dev)
            
            for d in datas:

                d = d.float().to(self.dev)
                
                # inference
                with torch.no_grad():
                    output, _ = self.model(d)
                    _, output_index = torch.max(output, 1)  
                    # print("prediction: ", output_index)
                
                    if evaluation:
                        for i in range(len(output_index)):
                            if (output_index[i] == label[i]):
                                correct +=1
                        total_data += len(output_index)

                    
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        
        return correct, (100*(correct/total_data))    # MTGEA freezing train 
       
    # MTGEA 2 stream train 
    def doubletrain(self, chunk=1):
        self.model.train()    
        self.adjust_lr()    
        loader = self.data_loader['first train']
        loader_2 = self.data_loader['second train']    
        loss_value = []
        loss_2_value = []
        loss_3_value = []

        for first, second in zip(loader, loader_2):

            firsts = first[0].chunk(chunk, dim=2)
            seconds = second[0].chunk(chunk, dim=2)
            for f, s in zip(firsts, seconds):

                f = f.float().to(self.dev)
                first[1] = first[1].long().to(self.dev)

                s = s.float().to(self.dev)
                second[1] = second[1].long().to(self.dev)

                #print("frist : " , f.shape, first[1].shape)
                #print("++++++++++++++++++++++++++")
                #print("second : " , s.shape, second[1].shape)
                #print("++++++++++++++++++++++++++")

                #forward
                output = self.model(f, s)
                loss_1 = self.loss_1(output, first[1])
                #kld loss 계산

                #backward
                self.optimizer.zero_grad()
                loss_1.backward(retain_graph=True)
                #kld backward
                self.optimizer.step()

                # statistics
                self.iter_info['loss'] = loss_1.data.item()
                self.iter_info['lr'] = '{:.6f}'.format(self.lr)
                loss_value.append(self.iter_info['loss'])
                self.show_iter_info()
                self.meta_info['iter'] += 1
            '''    
            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
            second[1] = second[1].long().to(self.dev)    


            # forward
            output  = self.model(first[0], second[0])
            loss_1 = self.loss_1(output, first[1])


            # backward
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss_1.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1
            '''

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()
 

        
    # MTGEA 2 stream train 
    def doubletest(self, evaluation=True, chunk=1):
        self.relu = torch.nn.ReLU()
        correct = 0
        total_data = 0
        self.model.eval()  
        loader = self.data_loader['first test']
        loader_2 = self.data_loader['second test']
        loss_value = []
        label_frag = []
        result_frag = []

        for first, second in zip(loader, loader_2):
            firsts = first[0].chunk(chunk, dim=2)
            seconds = second[0].chunk(chunk, dim=2)
            for f, s in zip(firsts, seconds):

                f = f.float().to(self.dev)
                first[1] = first[1].long().to(self.dev)

                s = s.float().to(self.dev)
                second[1] = second[1].long().to(self.dev)
                #print("frist : " , f.shape, first[1].shape)
                #print("++++++++++++++++++++++++++")
                #print("second : " , s.shape, second[1].shape)
                #print("++++++++++++++++++++++++++")
                # inference
                with torch.no_grad():
                    output = self.model(f, s)
                    _, final_output_index = torch.max(output, 1)
                    # print("prediction: ", final_output_index)


                if evaluation:
                    for i in range(len(final_output_index)):
                        if (final_output_index[i] == first[1][i]):
                            correct +=1
                    total_data += len(final_output_index)                 
                
            '''    
            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
    
            
            # print("test label: ", first[1])

            # inference
            with torch.no_grad():
                output = self.model(first[0], second[0])
                _, final_output_index = torch.max(output, 1)
                # print("prediction: ", final_output_index)


            if evaluation:
                for i in range(len(final_output_index)):
                    if (final_output_index[i] == first[1][i]):
                        correct +=1
                total_data += len(final_output_index) 
            '''
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        print("model parameter : ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))

        return correct, (100*(correct/total_data))   
    

    
    def freezingtrain(self, chunk=1):
        self.model.train()    
        self.adjust_lr()    
        loader = self.data_loader['first train']
        loader_2 = self.data_loader['second train']    
        loss_value = []
        for name, child in self.model.named_children():

            for param in child.parameters():
                if name == 'kin_stgcn':
                    param.requires_grad = False
 

        for first, second in zip(loader, loader_2):

            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
            second[1] = second[1].long().to(self.dev)    

            # forward
            output  = self.model(first[0], second[0])
            loss_1 = self.loss_1(output, first[1])

            # backward
            self.optimizer.zero_grad()
            loss_1.backward(retain_graph=True)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss_1.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])
            self.show_iter_info()
            self.meta_info['iter'] += 1

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    # MTGEA freezing test
    def freezingtest(self, evaluation=True, chunk=1):
        self.relu = torch.nn.ReLU()
        correct = 0
        total_data = 0
        self.model.eval()  
        loader = self.data_loader['first test']
        loader_2 = self.data_loader['second test']
        loss_value = []
        label_frag = []
        result_frag = []

        # check for parameter
        """
        for name, child in self.model.named_children():
            for param in child.parameters():
                if name == 'kin_stgcn':
                    print("kin freezing prarm", param)
        """            

        for first, second in zip(loader, loader_2):
    
            # get data
            first[0] = first[0].float().to(self.dev)
            first[1] = first[1].long().to(self.dev)

            second[0] = second[0].float().to(self.dev)
    
            
            # print("test label: ", first[1])

            # inference
            with torch.no_grad():
                output = self.model(first[0], second[0])
                _, final_output_index = torch.max(output, 1)
                # print("prediction: ", final_output_index)


            if evaluation:
                for i in range(len(final_output_index)):
                    if (final_output_index[i] == first[1][i]):
                        correct +=1
                total_data += len(final_output_index)       
        print("correct: ", correct)
        print("Accuracy: ", 100*(correct/total_data))
        print("==========================================") 
        
        return correct, (100*(correct/total_data))   


    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser   