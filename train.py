import os, time, warnings
import wandb
import torch
from ultralytics import RTDETR

# 关闭不必要警告
warnings.filterwarnings('ignore')

# ✅ 启动 wandb 实验
wandb.init(
    project='rtdetr-project',  # 可自定义你的 wandb 项目名
    name=time.strftime("rtdetr-%Y%m%d-%H%M%S"),
)

# ✅ 每个 epoch 结束时记录指标
def on_fit_epoch_end(trainer):
    metrics = trainer.metrics
    wandb.log({
        'epoch': trainer.epoch,
        'precision': metrics.get('metrics/precision(B)', 0),
        'recall': metrics.get('metrics/recall(B)', 0),
        'map50': metrics.get('metrics/mAP50(B)', 0),
        'map75': metrics.get('metrics/mAP75(B)', 0),
        'map50-95': metrics.get('metrics/mAP50-95(B)', 0),
    })

# ✅ 训练完成后记录模型统计信息
def log_static_model_info(model):
    params = sum(p.numel() for p in model.model.parameters()) / 1e6  # M

    try:
        flops = model.model.fuse().profile()[1] / 1e9  # G
    except:
        flops = 0

    model_path = 'runs/train/exp/weights/best.pt'
    size_mb = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0

    dummy = torch.zeros((1, 3, 640, 640)).to(model.device)
    t0 = time.time()
    model.predict(dummy, verbose=False)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)

    wandb.log({
        'params_M': round(params, 2),
        'flops_G': round(flops, 2),
        'model_size_MB': round(size_mb, 2),
        'fps': round(fps, 2),
    })

# ✅ 主函数入口
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')

    model.train(
        data='/root/autodl-tmp/neudet/data.yaml',
        cache=False,
        imgsz=640,
        epochs=200,
        batch=4,
        workers=4,
        project='runs/train',
        name=wandb.run.name,   # 实验名与 wandb 同步
        visualize=True,
        wandb=True,
        callbacks={'on_fit_epoch_end': on_fit_epoch_end},  # 注册回调
    )

    # ✅ 训练完成后记录模型静态信息
    log_static_model_info(model)
