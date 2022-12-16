#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import subprocess


def test() -> None:
    '''Run all unittests
    '''
    subprocess.run([
        'python', '-u', '-m',
        'pytest', '-v', '--cov'
    ])
    return
