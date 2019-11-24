# coding:utf-8
__author__ = 'frkhit'

from demo import call_complex_server, call_simple_server

if __name__ == '__main__':
    # call simple model
    call_simple_server(x=20.0)  # 20 * x + 3

    # call complex model
    call_complex_server(x=0, d=20)  # 20 * x + 3 + d
