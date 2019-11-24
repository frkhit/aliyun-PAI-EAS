# -*- coding:utf-8 -*-
import logging
import os
import zipfile
from urllib.parse import urlparse

import numpy as np
import tensorflow as tf
from pai_tf_predict_proto import tf_predict_pb2

from com.aliyun.api.gateway.sdk import client
from com.aliyun.api.gateway.sdk.common import constant
from com.aliyun.api.gateway.sdk.http import request


def zip_file(src_dir):
    zip_name = src_dir + '.zip'
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        for dirpath, dirnames, filenames in os.walk(src_dir):
            fpath = dirpath.replace(src_dir, '')
            fpath = fpath and fpath + os.sep or ''
            for filename in filenames:
                z.write(os.path.join(dirpath, filename), fpath + filename)


def get_last_meta_path(save_path):
    path = "/".join(save_path.split("/")[:-1])
    model_name = save_path.split("/")[-1]

    meta_file_info = {}
    for file_name in os.listdir(path):
        if file_name.find(model_name) == 0 and len(file_name) > 5 and file_name[-5:] == ".meta":
            step_str = file_name[:-5].split("-")[-1]
            try:
                meta_file_info[int(step_str)] = os.path.join(path, file_name)
            except ValueError as e:
                logging.error(e, exc_info=1)
                meta_file_info[0] = os.path.join(path, file_name)

    if not meta_file_info:
        return None

    meta_keys = list(meta_file_info.keys())
    meta_keys.sort()
    return meta_file_info[meta_keys[-1]]


def get_saver_and_last_step(meta_path, sess):
    if meta_path is None:
        return None, -1
    else:
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, meta_path[:-5])
        try:
            return saver, int(meta_path[:-5].split("-")[-1])
        except ValueError as e:
            logging.error(e, exc_info=1)
            return saver, -1


class LearningRate(object):
    def __init__(self):
        self._count = 0
        self._init = 0.01

    def get_learning_rate(self):
        return self._init * 0.95


class LinearFit(object):
    def __init__(self):
        self.sess = None
        self.learning_rate_manager = LearningRate()
        self.save_path = os.path.join(os.path.dirname(__file__), "models_meta", self.__class__.__name__)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        # 数据集
        train_data_size = 10000
        self.train_data_x = np.random.rand(train_data_size) * 10  # 0-10取值
        self.train_data_y = 20 * self.train_data_x + 3 + np.random.normal(loc=0, scale=0.1, size=(train_data_size,))
        self.test_data_x = np.arange(0, 10)
        self.test_data_y = 20 * self.test_data_x + 3

    @staticmethod
    def batch_data(x, y, size=128, last_cursor=None):
        if last_cursor is None:
            return x[:size], y[:size]
        else:
            if last_cursor + size >= x.shape[0]:
                return None, None
            return x[last_cursor: last_cursor + size], y[last_cursor:last_cursor + size]

    @staticmethod
    def build():
        # 参数
        tf_x = tf.placeholder(tf.float32, name="x")
        tf_y = tf.placeholder(tf.float32, name="y")
        tf_w = tf.Variable(0.0, name="w", )
        tf_b = tf.Variable(0.0, name="b", )
        tf_learning_rate = tf.Variable(0.01, name="learning_rate")

        tf_y_predict = tf.multiply(tf_x, tf_w) + tf_b

        cross_entropy = tf.reduce_mean(tf.multiply(tf.square(tf_y - tf_y_predict), 0.5))
        train_step = tf.train.GradientDescentOptimizer(tf_learning_rate).minimize(cross_entropy)
        tf.add_to_collection("inputs", tf_x)
        tf.add_to_collection("inputs", tf_y)
        tf.add_to_collection("outputs", tf_y_predict)
        tf.add_to_collection("outputs", cross_entropy)
        tf.add_to_collection("outputs", train_step)

    def train(self):
        if self.sess is None:
            self.sess = tf.InteractiveSession()

        saver, last_step = get_saver_and_last_step(get_last_meta_path(self.save_path), self.sess)
        if saver is None:
            # 没有持久化： 重新初始化模型
            print(" init models ...")
            self.build()
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            saver = tf.train.Saver()
        else:
            print(" restoring models ...")

        tf_x, tf_y = tf.get_collection('inputs')
        tf_y_predict, cross_entropy, train_step = tf.get_collection("outputs")
        graph = tf.get_default_graph()
        tf_w = graph.get_tensor_by_name("w:0")
        tf_b = graph.get_tensor_by_name("b:0")
        tf_learning_rate = graph.get_tensor_by_name("learning_rate:0")
        print("w is {}, b is {}".format(self.sess.run(tf_w), self.sess.run(tf_b)))

        batch_size = 1000
        global_step = last_step
        for i in range(10):
            train_data_cursor = 0
            while True:
                batch_x, batch_y = self.batch_data(self.train_data_x, self.train_data_y, batch_size, train_data_cursor)
                train_data_cursor = train_data_cursor + batch_size
                if batch_x is None and batch_y is None:
                    break
                self.sess.run(train_step, feed_dict={tf_x: batch_x,
                                                     tf_y: batch_y,
                                                     tf_learning_rate: self.learning_rate_manager.get_learning_rate()})

                global_step += 1
                if global_step % 10 == 0:
                    saver.save(self.sess, self.save_path, global_step=global_step)

        print("w is {}, b is {}".format(self.sess.run(tf_w), self.sess.run(tf_b)))
        print("cross is {}".format(self.sess.run(tf.reduce_mean(
            self.sess.run(cross_entropy, feed_dict={tf_x: self.test_data_x, tf_y: self.test_data_y})
        ))))

        self.sess.close()

    def build_simple_model(self, export_dir: str):
        """  """
        sess = tf.InteractiveSession()
        saver, last_step = get_saver_and_last_step(get_last_meta_path(self.save_path), sess)
        tf_x, tf_y = tf.get_collection('inputs')
        tf_y_predict, cross_entropy, train_step = tf.get_collection("outputs")
        graph = tf.get_default_graph()
        tf_w = graph.get_tensor_by_name("w:0")
        tf_b = graph.get_tensor_by_name("b:0")
        tf.saved_model.simple_save(
            session=sess,
            export_dir=export_dir,
            inputs={"x": tf_x},
            outputs={"y": tf_y_predict},
        )
        sess.close()

    def build_complex_model(self, export_dir: str):
        """  """
        sess = tf.InteractiveSession()
        saver, last_step = get_saver_and_last_step(get_last_meta_path(self.save_path), sess)
        tf_x, tf_y = tf.get_collection('inputs')
        tf_y_predict, cross_entropy, train_step = tf.get_collection("outputs")
        graph = tf.get_default_graph()
        tf_w = graph.get_tensor_by_name("w:0")
        tf_b = graph.get_tensor_by_name("b:0")

        # 调整模型
        tf_d = tf.placeholder(tf.float32, name="d")
        new_y = tf_y_predict + tf_d

        tf.saved_model.simple_save(
            session=sess,
            export_dir=export_dir,
            inputs={"x": tf_x, "d": tf_d},
            outputs={"y": new_y},
        )
        sess.close()

    def serving(self, saved_model_dir: str):
        """ 运行服务 """
        pass


class PAIClientDemo(object):
    app_key = 'xxx'
    app_secret = 'xxx'

    @staticmethod
    def predict(url, app_key, app_secret, request_data):
        cli = client.DefaultClient(app_key=app_key, app_secret=app_secret)
        body = request_data
        url_ele = urlparse(url)
        host = 'https://' + url_ele.hostname
        path = url_ele.path
        req_post = request.Request(host=host, protocol=constant.HTTP, url=path, method="POST", time_out=6000)
        req_post.set_body(body)
        req_post.set_content_type(constant.CONTENT_TYPE_STREAM)
        stat, header, content = cli.execute(req_post)
        return stat, dict(header) if header is not None else {}, content

    def simple(self, x: float):
        # 输入模型信息,点击模型名字就可以获取到了
        url = 'https://xxxx-cn-shenzhen.alicloudapi.com/EAPI_1372988890346240_demo_simple'

        # 构造服务
        _request = tf_predict_pb2.PredictRequest()
        _request.signature_name = 'serving_default'
        _request.inputs['x'].dtype = tf_predict_pb2.DT_FLOAT  # images 参数类型
        _request.inputs['x'].float_val.extend([x])

        # 将pb序列化成string进行传输
        request_data = _request.SerializeToString()
        stat, header, content = self.predict(url, self.app_key, self.app_secret, request_data)
        if stat != 200:
            print('Http status code: ', stat)
            print('Error msg in header: ', header['x-ca-error-message'] if 'x-ca-error-message' in header else '')
            print('Error msg in body: ', content)
        else:
            response = tf_predict_pb2.PredictResponse()
            response.ParseFromString(content)
            print(response)

    def complex(self, x: float, d: float):
        # 输入模型信息,点击模型名字就可以获取到了
        url = "https://xxxx-cn-shenzhen.alicloudapi.com/EAPI_1372988890346240_demo_complex"

        # 构造服务
        _request = tf_predict_pb2.PredictRequest()
        _request.signature_name = 'serving_default'
        _request.inputs['x'].dtype = tf_predict_pb2.DT_FLOAT  # images 参数类型
        _request.inputs['x'].float_val.extend([x])

        _request.inputs['d'].dtype = tf_predict_pb2.DT_FLOAT  # images 参数类型
        _request.inputs['d'].float_val.extend([d])

        # 将pb序列化成string进行传输
        request_data = _request.SerializeToString()
        stat, header, content = self.predict(url, self.app_key, self.app_secret, request_data)
        if stat != 200:
            print('Http status code: ', stat)
            print('Error msg in header: ', header['x-ca-error-message'] if 'x-ca-error-message' in header else '')
            print('Error msg in body: ', content)
        else:
            response = tf_predict_pb2.PredictResponse()
            response.ParseFromString(content)
            print(response)


def build_model(_export_dir: str):
    if not os.path.exists(_export_dir):
        os.makedirs(_export_dir, exist_ok=True)

    LinearFit().build_simple_model(export_dir=_export_dir)

    zip_file(_export_dir)


def call_simple_server(x: float):
    PAIClientDemo().simple(x=x)


def build_complex_model(_export_dir: str):
    if not os.path.exists(_export_dir):
        os.makedirs(_export_dir, exist_ok=True)

    LinearFit().build_complex_model(export_dir=_export_dir)

    zip_file(_export_dir)


def call_complex_server(x: float, d: float):
    PAIClientDemo().complex(x=x, d=d)
