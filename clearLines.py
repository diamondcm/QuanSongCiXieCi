#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import collections
import logging
import random

import numpy as np
import json
from flags import parse_args

def clearBlankLine():
    file1 = open('QuanSongCi_.txt', 'r', encoding='utf-8') # 要去掉空行的文件 
    file2 = open('QuanSongCi_noLine.txt', 'w', encoding='utf-8') # 生成没有空行的文件
    try:
        for line in file1.readlines():
            if line == '\n':
                line = line.strip("\n")
            file2.write(line)
    finally:
        file1.close()
        file2.close()


if __name__ == '__main__':
    clearBlankLine()
