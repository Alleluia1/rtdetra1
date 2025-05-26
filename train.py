import os, time, warnings
import wandb
import torch
from ultralytics import RTDETR

# ✅ 自动登录 WandB（替代手动 login）
os.environ['WANDB_API_KEY'] = 'f8cb8b13b090d70eb2b9b5ee36da161979b90a95'

# 关闭警告
warnings.filterwarnings('ignore')

# ✅ 初始化 WandB 项目
wandb.init(
    project='rtdetr-project',
    name=time.strftime("rtdetr-%Y%m%d-%H%M%S"),
)

# ✅ 每个 epoch 结束时记录 loss（兼容 RT-DETR）
def on_fit_epoch_end(trainer):
    # 建议打印一次查看结构（调试用）
    # print("DEBUG: trainer.__dict__ =", trainer.__dict__)
    try:
        loss_dict = trainer.loss
        if isinstance(loss_dict, dict):
            wandb.log({
                'epoch': trainer.epoch,
                'loss_cls': loss_dict.get('loss_cls', 0),
                'loss_box': loss_dict.get('loss_bbox', 0),
                'loss_obj': loss_dict.get('loss_obj', 0),
                'loss_total': sum(loss_dict.values())
            })
        else:
            wandb.log({'epoch': trainer.epoch, 'loss': loss_dict})
    except Exception as e:
        print(f"⚠️ WandB 日志记录失败: {e}")

# ✅ 训练结束后记录模型信息
def log_static_model_info(model):
    params = sum(p.numel() for p in model.model.parameters()) / 1e6  # 参数量 M
    model_path = 'runs/train/exp/weights/best.pt'
    size_mb = os.path.getsize(model_path) / 1e6 if os.path.exists(model_path) else 0
    dummy = torch.zeros((1, 3, 640, 640)).to(model.device)
    t0 = time.time()
    model.predict(dummy, verbose=False)
    t1 = time.time()
    fps = 1.0 / (t1 - t0)

    wandb.log({
        'params_M': round(params, 2),
        'model_size_MB': round(size_mb, 2),
        'fps': round(fps, 2),
    })

# ✅ 主程序入口
if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')

    model.train(
        data='/root/autodl-tmp/neudet/data.yaml',
        imgsz=640,
        epochs=200,
        batch=4,
        workers=4,
        project='runs/train',
        name=wandb.run.name,
        visualize=True,
        callbacks={'on_fit_epoch_end': on_fit_epoch_end},  # 注册 WandB 回调
    )

    # ✅ 训练后记录模型信息（如存在 best.pt）
    if os.path.exists('runs/train/exp/weights/best.pt'):
        log_static_model_info(model)
    else:
        print("⚠️ 未找到 best.pt，跳过模型信息记录")
