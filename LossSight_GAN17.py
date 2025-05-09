# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from gain17 import gain
from topo import topo

def main (args):

  '''
  Args:
    - data_name: LossSight
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    - num_motif: number of motifs
    - order : order of polynomial
    - step_size: step_size
    - filters: num of filters in GLU
    - kernel_size : size of kernel in GLU 
    - learning_rate : learning_rate
    - num_columns_of_gru : num_columns_of_gru

  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  data_name = args.data_name
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'num_motif': args.num_motif,
                     'order':args.order,
                     'step_size':args.step_size,
                     'filters':args.filters,
                     'kernel_size':args.kernel_size,
                     'learning_rate':args.learning_rate,
                     'num_columns_of_gru':args.num_columns_of_gru}
  
  file_name = 'data/'+data_name+'.csv'
  miss_data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  full_data_x = np.loadtxt('data/queue_depth_data_full.csv', delimiter=",", skiprows=1)
  # Create or load adjacency matrices A_list
  
  
  
  #A = np.eye(miss_data_x.shape[0])
  A_list = topo()
  #A_list = [A for _ in range(args.num_motif)]  # Example of random adjacency matrices

  # Call gain function with the loaded data, adjacency matrices and parameters
  imputed_train_data, imputed_test_data = gain(miss_data_x, full_data_x, gain_parameters, A_list)


  # Save Result
  np.savetxt("result10_train.csv", imputed_train_data, delimiter=',',fmt='%.1f')
  np.savetxt("result10_test.csv", imputed_test_data, delimiter=',',fmt='%.1f')

  return imputed_train_data, imputed_test_data

if __name__ == '__main__':  
  
  # Inputs for the main function
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      default='LossSight',
      type=str)
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=128,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10000,
      type=int)
  parser.add_argument(
      '--num_motif',
      help='number of motifs',
      default=16,
      type=int)
  parser.add_argument(
      '--order',
      help='order of polynomial',
      default=1,
      type=int)
  parser.add_argument(
      '--step_size',
      help='step_size',
      default=32,
      type=int)
  parser.add_argument(
      '--filters',
      help='num of filters in GLU',
      default=3,
      type=int)
  parser.add_argument(
      '--kernel_size',
      help='size of kernel in GLU ',
      default=4,
      type=int)
  parser.add_argument(
      '--learning_rate',
      help='learning_rate',
      default=0.0001,
      type=float)
  parser.add_argument(
      '--num_columns_of_gru',
      help='num_columns_of_gru',
      default=16,
      type=int)
  args = parser.parse_args() 
  
  # Calls main function  
  imputed_data = main(args)
