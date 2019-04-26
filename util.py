#!/usr/bin/env python
# encoding: utf-8

def get_args(parser):
    parser.add_argument('--lamb', type=float, default=1, help='value of regularizing constant')
    parser.add_argument('--K', type=int, default=10, help='size of prior distribution')
    parser.add_argument('--dim', type=int, default=10, help='dimension of action space')
    parser.add_argument('--expstart', type=int, default=1, help='numerotation start')
    parser.add_argument('--nexp', type=int, default=1, help='number of experiments per method')
    parser.add_argument('--sinkiter', type=int, default=10, help='number of iterations per sinkhorn loop')
    parser.add_argument('--sinklr', type=float, default=1, help='learning rate for sinkhorn method')
    parser.add_argument('--descentlr', type=float, default=0.1, help='learning rate for descent method')    
    parser.add_argument('--dclr', type=float, default=0.0001, help='learning rate for dc method')    
    parser.add_argument('--dcdualiter', type=int, default=10, help='number of dual iterations per dc iteration')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--verbose_freq', type=int, default=int(1e6), help='print current number of iterations every verbose_freq iterations')
    parser.add_argument('--dcmaxiter', type=int, default=1000, help='number of iterations for dc method')
    parser.add_argument('--sinkmaxiter', type=int, default=1000, help='number of iterations for sinkhorn method')
    parser.add_argument('--descentmaxiter', type=int, default=10000, help='number of iterations for descent method')
    parser.add_argument('--gpu', type=bool, default=False, help='If set to True, train with GPU')
    return parser