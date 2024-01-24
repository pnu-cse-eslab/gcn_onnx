#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np
from net.st_gcn_alone import alone
import random
# torch
import torch
import torch.nn as nn
import torch.optim as optim

import onnxruntime
from onnxruntime.quantization import QuantFormat, QuantType, StaticQuantConfig, quantize, CalibrationMethod
from onnxruntime.quantization import CalibrationDataReader

# torchlight
import torchlight
import time
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .io import IO

# input_model_path = 'my_model-sim.onnx'
input_model_path = 'aqw.onnx'
output_model_path = 'aqw_quantized.onnx'

class XXXDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, loader):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, chnannel, height, width) = session.get_inputs()[0].shape

        # print([data[..., 0].numpy().astype(np.float32).shape for data, _ in loader])
        
        # Generate the random data in the half-open interval [0.0, 1.0).
        self.nhwc_data_list = [data.numpy().astype(np.float32) for data, _ in loader]

        np.save('calib_img.npy', np.concatenate([data.numpy().astype(np.float32) for data, _ in loader], axis=0))
        
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
        
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    

def benchmark(model_path, loader):
    session = onnxruntime.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name

    total = 0.0
    input_data = np.zeros((1, 3, 200, 25), np.float32)
    # Warming up
    _ = session.run([], {input_name: input_data})

    print(loader)
    t = 0
    tp = 0

    # for i in range(runs):
    for data, label in loader:
        t += 1
        # print(data.shape)
        start = time.perf_counter()
        if label.numpy()[0] == np.argmax(session.run([], {input_name: data.numpy().astype(np.float32)})[0]):
            tp += 1
        end = (time.perf_counter() - start) * 1000
        total += end
        # print(f"{end:.2f}ms")
    total /= t
    print(f"Avg: {total:.2f}ms")
    print(f"Acc: {tp / t * 100}%")

    np.save('input_sample.npy', data.numpy().astype(np.float32))



class Processor(IO):
    """
        Base Processor
    """

    def __init__(self, argv=None):

        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()
        if self.arg.kd_model != 'null':
            self.kd_model = alone(base_model='net.st_gcn_128.Model', output_class=7, in_channels=3, num_class=32, edge_importance_weighting=True, graph_args={'layout':'ntu-rgb+d', 'strategy':'spatial'})
            self.kd_model.load_state_dict(torch.load(self.arg.kd_model))
            self.kd_model.cuda()
        else:
            self.kd_model = "null"
        self.best_acc = -1
        self.seed = 25872587

    def init_environment(self):

        super().init_environment()
        self.result = dict()
        self.iter_info = dict()
        self.epoch_info = dict()
        self.meta_info = dict(epoch=0, iter=0)

    def load_optimizer(self):
        pass


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        if 'debug' not in self.arg.train_feeder_args:
            self.arg.train_feeder_args['debug'] = self.arg.debug
        self.data_loader = dict()
        if self.arg.phase == 'train' :
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)
        if self.arg.test_feeder_args:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))                
        if self.arg.phase == 'double_train' or self.arg.phase == 'kd_train':                                        # MTGEA 2 stream train 
            set_seed(200)
            self.data_loader['first train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True,worker_init_fn=np.random.seed(0))       
            set_seed(200)
            self.data_loader['second train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.skeleton_train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True, worker_init_fn=np.random.seed(0))
        if self.arg.phase == 'freezing_train':                                      # MTGEA 2 freezing train 
            self.data_loader['first train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)       
            self.data_loader['second train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.skeleton_train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device),
                drop_last=True)                              
        if self.arg.skeleton_test_feeder_args:                                      
            self.data_loader['first test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))
            self.data_loader['second test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.skeleton_test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(
                    self.arg.device))



    def show_epoch_info(self):
        for k, v in self.epoch_info.items():
            self.io.print_log('\t{}: {}'.format(k, v))
        if self.arg.pavi_log:
            self.io.log('train', self.meta_info['iter'], self.epoch_info)

    def show_iter_info(self):
        if self.meta_info['iter'] % self.arg.log_interval == 0:
            info ='\tIter {} Done.'.format(self.meta_info['iter'])
            for k, v in self.iter_info.items():
                if isinstance(v, float):
                    info = info + ' | {}: {:.4f}'.format(k, v)
                else:
                    info = info + ' | {}: {}'.format(k, v)

            self.io.print_log(info)

            if self.arg.pavi_log:
                self.io.log('train', self.meta_info['iter'], self.iter_info)

    def train(self):
        for _ in range(100):
            self.iter_info['loss'] = 0
            self.show_iter_info()
            self.meta_info['iter'] += 1
        self.epoch_info['mean loss'] = 0
        self.show_epoch_info()

    def test(self):
        for _ in range(100):
            self.iter_info['loss'] = 1
            self.show_iter_info()
        self.epoch_info['mean loss'] = 1
        self.show_epoch_info()

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
        #summary(self.model, input_size=(3, 400, 25))
        print(self.model)
        # training phase
        
        if self.arg.phase == 'train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.train(chunk=self.arg.chunk)
                self.io.print_log('Done.')
                self.io.print_log('Eval epoch: {}'.format(epoch)) 
                
                cor, acc = self.slidingtest(chunk=self.arg.chunk)
                if self.best_acc < acc:
                    file_text = str(cor) + "_" + str(acc) + '_epoch{}_model.pt'.format(epoch + 1)  
                    filename = file_text        # 2
                    self.io.save_model(self.model, filename)
                    self.io.print_log('Done.')
                    self.best_acc = acc
                elif self.best_acc == 100:
                    break
                else:
                    self.io.print_log('Not update.')

        # test phase
        elif self.arg.phase == 'test' :

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.test(chunk=self.arg.chunk)
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')
            
            Feeder = import_class(self.arg.feeder)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                            dataset=Feeder(**self.arg.train_feeder_args),
                            batch_size=self.arg.batch_size,
                            shuffle=True,
                            num_workers=self.arg.num_worker * torchlight.ngpu(
                                self.arg.device),
                            drop_last=True)
            
            train_loader = self.data_loader['train']
            test_loader = self.data_loader['test']

            for i, (data, label) in enumerate(test_loader):
                data[..., 0].numpy().astype(np.float32).tofile(f'z/{label.numpy()[0]}/{i}', format='f')

            torch.onnx.export(
                self.model.to(self.dev),
                torch.randn(1, 3, 200, 25).float().to(self.dev),
                'my_model.onnx',
                do_constant_folding=True,
                opset_version=13,
                export_params=True,
                verbose=False,
            )
            '''
            dr = XXXDataReader(input_model_path, train_loader)

            conf = StaticQuantConfig(
                calibration_data_reader=dr,
                quant_format=QuantFormat.QDQ,
                calibrate_method=CalibrationMethod.MinMax,
                optimize_model=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True)
                
            quantize(input_model_path, output_model_path, conf)

            print("benchmarking fp32 model...")
            benchmark(input_model_path, test_loader)

            print("benchmarking int8 model...")
            benchmark(output_model_path, test_loader)
            '''
            
            
        # MTGEA 2 stream train
        elif self.arg.phase =='onnx':

            
            
            
            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')
            
            Feeder = import_class(self.arg.feeder)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                            dataset=Feeder(**self.arg.train_feeder_args),
                            batch_size=self.arg.batch_size,
                            shuffle=True,
                            num_workers=self.arg.num_worker * torchlight.ngpu(
                                self.arg.device),
                            drop_last=True)
            
            train_loader = self.data_loader['train']
            test_loader = self.data_loader['test']

            for i, (data, label) in enumerate(test_loader):
                data.numpy().astype(np.float32).tofile(f'z/{label.numpy()[0]}/{i}', format='f')

            torch.onnx.export(
                self.model.to(self.dev),
                torch.randn(1, 3, 100, 25).float().to(self.dev),
                'aqw.onnx',
                do_constant_folding=True,
                opset_version=14,
                export_params=True,
                verbose=False,
            )
            
            dr = XXXDataReader(input_model_path, test_loader)

            conf = StaticQuantConfig(
                calibration_data_reader=dr,
                quant_format=QuantFormat.QDQ,
                calibrate_method=CalibrationMethod.MinMax,
                optimize_model=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True)
                
            quantize(input_model_path, output_model_path, conf)

            print("benchmarking fp32 model...")
            benchmark(input_model_path, test_loader)

            print("benchmarking int8 model...")
            benchmark(output_model_path, test_loader)
            
        elif self.arg.phase == 'double_train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch
                # training
                self.io.print_log('Training epoch: {}'.format(epoch))       
                self.doubletrain(chunk=self.arg.chunk)  
                self.io.print_log('Done.')
                self.io.print_log('Eval epoch: {}'.format(epoch))        
                cor, acc = self.doubletest()                                                     
                file_text = str(cor) + "_" + str(acc) + '_'+'epoch{}_model.pt'.format(epoch + 1)                                                  
                # file_text =  '_'+'epoch{}_model.pt'.format(epoch + 1)                   
                filename = file_text       
                self.io.save_model(self.model, filename)   
                self.io.print_log('Done.')

        # MTGEA 2 stream test
        elif self.arg.phase == 'double_test':
            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))
            self.io.print_log('Evaluation Start:')
            self.doubletest(chunk=self.arg.chunk)
            self.io.print_log('Done.\n')
    
            # # save the output of model
            # if self.arg.save_result:
            #     result_dict = dict(
            #         zip(self.data_loader['test'].dataset.sample_name,
            #             self.result))
            #     self.io.save_pkl(result_dict, 'test_result.pkl')    

        # MTGEA 2 freezing train 
        elif self.arg.phase == 'freezing_train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))        
                self.freezingtrain(chunk=self.arg.chunk)  
                self.io.print_log('Done.')
                self.io.print_log('Eval epoch: {}'.format(epoch))        
                cor, acc = self.freezingtest()                                                     
                file_text = str(cor) + "_" + str(acc) + '_'+'epoch{}_model.pt'.format(epoch + 1)
                filename = file_text       
                self.io.save_model(self.model, filename)   
                self.io.print_log('Done.')

        elif self.arg.phase == 'slidingtest':

            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))

            # evaluation
            self.io.print_log('Evaluation Start:')
            self.slidingtest(chunk=self.arg.chunk)
            self.io.print_log('Done.\n')

            # save the output of model
            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader['test'].dataset.sample_name,
                        self.result))
                self.io.save_pkl(result_dict, 'test_result.pkl')
            
            Feeder = import_class(self.arg.feeder)

            self.data_loader['train'] = torch.utils.data.DataLoader(
                            dataset=Feeder(**self.arg.train_feeder_args),
                            batch_size=self.arg.batch_size,
                            shuffle=True,
                            num_workers=self.arg.num_worker * torchlight.ngpu(
                                self.arg.device),
                            drop_last=True)
            
            train_loader = self.data_loader['train']
            test_loader = self.data_loader['test']

            for i, (data, label) in enumerate(test_loader):
                data[..., 0].numpy().astype(np.float32).tofile(f'z/{label.numpy()[0]}/{i}', format='f')

            torch.onnx.export(
                self.model.to(self.dev),
                torch.randn(1, 3, 200, 25).float().to(self.dev),
                'my_model.onnx',
                do_constant_folding=True,
                opset_version=13,
                export_params=True,
                verbose=False,
            )


        # MTGEA 2 freezing test
        elif self.arg.phase == 'freezing_test':
            # the path of weights must be appointed
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.io.print_log('Model:   {}.'.format(self.arg.model))
            self.io.print_log('Weights: {}.'.format(self.arg.weights))
            # evaluation
            self.io.print_log('Evaluation Start:')
            self.freezingtest()
            self.io.print_log('Done.\n')
        


        elif self.arg.phase == 'kd_train':
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info['epoch'] = epoch

                # training
                self.io.print_log('Training epoch: {}'.format(epoch))
                self.kd_train(chunk=self.arg.chunk)
                self.io.print_log('Done.')
                self.io.print_log('Eval epoch: {}'.format(epoch)) 
                
                cor, acc = self.slidingtest(chunk=self.arg.chunk)
                if self.best_acc < acc:
                    file_text = str(cor) + "_" + str(acc) + '_epoch{}_model.pt'.format(epoch + 1)  
                    filename = file_text        # 2
                    self.io.save_model(self.model, filename)
                    self.io.print_log('Done.')
                    self.best_acc = acc
                elif self.best_acc == 100:
                    break
                else:
                    self.io.print_log('Not update.')
        self.io.print_log("**********************************************")
        self.io.print_log("model parameter : {}".format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    @staticmethod
    def get_parser(add_help=False):

        #region arguments yapf: disable
        # parameter priority: command line > config > default
        parser = argparse.ArgumentParser( add_help=add_help, description='Base Processor')

        parser.add_argument('-w', '--work_dir', default='./work_dir/tmp', help='the work folder for storing results')
        parser.add_argument('-c', '--config', default=None, help='path to the configuration file')
        parser.add_argument('--chunk', type=int, default=1, help='chunk size')

        # processor
        parser.add_argument('--phase', default='train', help='must be train or test')
        parser.add_argument('--save_result', type=str2bool, default=False, help='if ture, the output of the model will be stored')
        parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
        parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
        parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPUs or not')
        parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')

        # visulize and debug
        parser.add_argument('--log_interval', type=int, default=100, help='the interval for printing messages (#iteration)')
        parser.add_argument('--save_interval', type=int, default=10, help='the interval for storing models (#iteration)')
        parser.add_argument('--eval_interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
        parser.add_argument('--save_log', type=str2bool, default=True, help='save logging or not')
        parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
        parser.add_argument('--pavi_log', type=str2bool, default=False, help='logging on pavi or not')

        # feeder
        parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
        parser.add_argument('--num_worker', type=int, default=4, help='the number of worker per gpu for data loader')
        parser.add_argument('--train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        
        #for KD model
        parser.add_argument('--kd_model', type=str, default="null", help='KD model path')
        #### add function ####
        parser.add_argument('--skeleton_train_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for training')
        parser.add_argument('--skeleton_test_feeder_args', action=DictAction, default=dict(), help='the arguments of data loader for test')
        #### add function ####
        parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
        parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
        parser.add_argument('--debug', action="store_true", help='less data, faster loading')

        # model
        parser.add_argument('--model', default=None, help='the model will be used')
        parser.add_argument('--model_args', action=DictAction, default=dict(), help='the arguments of model')
        parser.add_argument('--weights', default=None, help='the weights for network initialization')
        parser.add_argument('--ignore_weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')
        #endregion yapf: enable

        return parser
