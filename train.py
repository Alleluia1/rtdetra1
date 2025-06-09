<<<<<<< HEAD
import warnings, os
os.environ['WANDB_MODE'] = 'offline' 
os.environ['WANDB_API_KEY'] = 'f8cb8b13b090d70eb2b9b5ee36da161979b90a95'

import time
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    # 代表用cpu训练 不推荐！没意义！ 而且有些模块不能在cpu上跑
# os.environ["CUDA_VISIBLE_DEVICES"]="0"     # 代表用第一张卡进行训练  0：第一张卡 1：第二张卡
# 多卡训练参考<使用教程.md>下方常见错误和解决方案
=======
import warnings, os, time
os.environ['WANDB_API_KEY'] = 'f8cb8b13b090d70eb2b9b5ee36da161979b90a95'  # ⚠️ 仅用于测试时使用环境变量，不推荐硬编码

# 启用 wandb 在线模式（可选）
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_INIT_TIMEOUT"] = "180"

>>>>>>> c867373d1d10a5bc769d5a2398469f0649a5e911
warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import wandb  # ← 加上这一行导入 wandb

if __name__ == '__main__':
    run = wandb.init(project="redet_backbone", name=time.strftime("%Y-%m-%d_%H-%M-%S"))  # ← 初始化 wandb 项目
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')
<<<<<<< HEAD
    # model.load('') # loading pretrain weights
    model.train(data='/root/autodl-tmp/neudet/data.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4, # batchsize 不建议乱动，一般来说4的效果都是最好的，越大的batch效果会很差(经验之谈)
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<使用教程.md>下方常见错误和解决方案
                # resume='', # last.pt path
                project='rtdetra1',
                name = time.strftime("%Y-%m-%d_%H-%M-%S"),
                )

=======

    # 启动训练
    model.train(
        data='E:/2345Downloads/Alleluia/redetr/redet_backbone/dataset/data.yaml',
        cache=False,
        imgsz=640,
        epochs=200,
        batch=4,
        workers=4,
        project='rtdetr_backbone',
        name=run.name  # 与 wandb run 同名，方便追踪
    )

    # 上传模型权重（默认保存在 runs/train/<exp>/weights 下）
    weight_dir = f'runs/rtdetr_backbone/{run.name}/weights'
    wandb.save(f'{weight_dir}/best.pt')
    wandb.save(f'{weight_dir}/last.pt')

    run.finish()  # 🧹 清理和关闭 wandb 运行
>>>>>>> c867373d1d10a5bc769d5a2398469f0649a5e911
