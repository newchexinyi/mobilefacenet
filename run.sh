# set tsinghua source
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# install package
pip install -r requirements.txt
# run
# train model
python run.py --cuda=True --device_ids=0,1 --batch_size=128 --arcface=True
# test best model
python run.py --load_epoch=best --match_factor=3 --cuda=True --device_ids=0,1 --batch_size=128
#visualization
tensorboard --logdir=runs