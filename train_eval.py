#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os


from flags import parse_args
from flags import train_times


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()
    print('current working dir [{0}]'.format(os.getcwd()))
    w_d = os.path.dirname(os.path.abspath(__file__))
    print('change wording dir to [{0}]'.format(w_d))
    os.chdir(w_d)

    cmd = ""
    for parm in ["output_dir", "text", "num_steps", "batch_size", "dictionary", "reverse_dictionary", "learning_rate"]:
        try:
            cmd += ' --{0}={1}'.format(parm, getattr(FLAGS, parm))
        except:
            pass

    for i in range(300):
       # train 1 epoch
        print('################    train  [{0}]  ################'.format(i))
        p = os.popen('python ./train.py' + cmd)
        for l in p:
            print(l.strip())
        train_times += 1
  
        # eval
        print('################    eval    ################')
        p = os.popen('python ./sample.py' + cmd)
        for l in p:
            print(l.strip())

