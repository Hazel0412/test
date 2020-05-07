import argparse
import sys

param_dict = {"batch_size": 50,                                    #  DataLoader将数据分割成mini-batch，每个mini-batch的大小
             "num_stack":2,                                        #  LSTM纵向堆叠的层数
             "preprocess":True,                                    #  是否对输入LSTM的梯度进行预处理，默认为True
             "p":10,                                               #  对输入LSTM的梯度预处理的判据，默认为10
             "output_scale":0.1,                                   #  对LSTM输出的梯度更新值进行缩放，默认为0.1

             "USE_CUDA":True,                                      #  是否使用GPU，默认为True
             "LSTM_TRAIN_ITERATION":100,                           #  训练LSTM参数的迭代次数，默认为100
             "UNROLL_ITERATION":20,                                #  训练LSTM一次需要原问题提供多少轮梯度信息，默认为20
             "THETA_RESET_INTERVAL":10,                            #  为了避免LSTM对原问题参数过拟合，间隔多少次对参数进行重置，默认为10
             "LSTM_ADAM_LR":0.001,                                 #  使用ADAM算法对LSTM优化，学习率是多少，默认为0.001
             "LSTM_ADAM_BETAS":(0.9,0.999),                        #  使用ADAM算法对LSTM优化，计算梯度平均的系数是多少，默认为(0.9, 0.999)
             "LSTM_ADAM_EPS":1e-8,                                 #  使用ADAM算法对LSTM优化，增强数值稳定性需要的附加项，默认为1e-8
             "LSTM_ADAM_WD":0,                                     #  使用ADAM算法对LSTM优化，weight_decay大小，默认为0
            }

parser = argparse.ArgumentParser(description='Parameters for learning to learn')

parser.add_argument('--USE_CUDA', action='store_true', default=True,
                    help='enables CUDA training')
parser.add_argument('--LSTM_TRAIN_ITERATION', type=int, default=100, metavar='N',
                    help='# of meta optimizer steps (default: 100)')
parser.add_argument('--UNROLL_ITERATION', type=int, default=20, metavar='N',
                    help='# of iterations for loss function (default: 20)')
parser.add_argument('--THETA_RESET_INTERVAL', type=int, default=10, metavar='N',
                    help='# of iterations to reset parameters of optimizee (default: 10)')
parser.add_argument('--LSTM_ADAM_LR', type=float, default=0.001, metavar='N',
                    help='learning rate for ADAM (default: 0.001)')
parser.add_argument('--LSTM_ADAM_BETAS', type=float, default=0.9, metavar='N',
                    help='coefficient for averaging gradient (default: 0.9)')
parser.add_argument('--LSTM_ADAM_EPS', type=float, default=1e-8, metavar='N',
                    help='additional param for stability (default: 1e-8)')
parser.add_argument('--LSTM_ADAM_WD', type=int, default=0, metavar='N',
                    help='weight_decay (default: 0)')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--num_stack', type=int, default=2, metavar='N',
                    help='# of LSTM layers (default: 2)')
parser.add_argument('--preprocess', action='store_true', default=True,
                    help='enables LSTM preprocess')
parser.add_argument('--p', type=int, default=10, metavar='N',
                    help='criterion for preprocess of gradient (default: 10)')
parser.add_argument('--output_scale', type=float, default=0.1, metavar='N',
                    help='scale for updates of gradient (default: 0.1)')

args = parser.parse_args()
print(args.batch_size)

