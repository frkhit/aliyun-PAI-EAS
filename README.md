# PAI-EAS 试用 DEMO

## 环境安装

```
python -m pip install -r requirements.txt
```

## 部署到PAI-EAS

- 训练模型并导出 SavedModel: `python train.py train`, `python train.py simple`, `python train.py complex`
- 将 `demo_simple.zip` 及 `demo_complex.zip` 分别上传到 PAI-EAS
- 分别获取两个模型的公网访问地址及授权码, 复制到`demo.py`中
- 远程调用 PAI-EAS: `python call_pai_eas.py`

## 感谢

- api网关签名算法参考了 [williezh/api-gateway-demo-sign-python](https://github.com/williezh/api-gateway-demo-sign-python)
