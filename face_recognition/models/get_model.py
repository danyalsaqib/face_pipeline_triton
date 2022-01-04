import subprocess
import os

os.chdir("face_recognition/models")
# Downloading pretrained models
if os.path.isfile('arcface.onnx'):
  print("File 'arcface.onnx' already exists")
else:
  subprocess.run(['gdown', '--id', '16aJ_uiDWeggv0V7i9VBSzrdIOUhK84Qr'])

if os.path.isfile('onet.onnx'):
  print("File 'onet.onnx' already exists")
else:
  subprocess.run(['gdown', '--id', '1ESxGRs96gxwp0LT9K1lpU9BzLATidmvS'])

if os.path.isfile('rnet.onnx'):
  print("File 'rnet.onnx' already exists")
else:
  subprocess.run(['gdown', '--id', '1JvcVrFETQ9YOaMJ_ce5H4vtkBeo_orZs'])

if os.path.isfile('pnet.onnx'):
  print("File 'pnet.onnx' already exists")
else:
  subprocess.run(['gdown', '--id', '1ZOJaIQGHxDIxErYiReB09tSYRF793POz'])

os.chdir("../..")
