import os, time, warnings
import wandb
import torch
from ultralytics import RTDETR

# ✅ 自动登录 WandB，无需手动输入
os.environ['WANDB_API_KEY'] = 'f8cb8b13b090d70eb2b9b5ee36da161979b90a95'

warnings.filterwarnings('ignore')

wandb.init(
    project='rtdetr-project',
    name=time.strftime("rtdetr-%Y%m%d-%H%M%S"),
)

# ✅ 2. 每个 epoch 结束时手动记录 loss
def on_fit_epoch_end(trainer):
    # 通常 loss_items: [cls_loss, box_loss, obj_loss, total_loss]
    if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
        wandb.log({
            'epoch': trainer.epoch,
            'loss_cls': trainer.loss_items[0],
            'loss_box': trainer.loss_items[1],
            'loss_obj': trainer.loss_items[2],
            'loss_total': sum(trainer.loss_items),
        })
    else:
        print("⚠️ trainer.loss_items 不存在或为空")

# ✅ 3. 训练完成后记录模型静态信息（可选）
def log_static_model_info(model):
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
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

# ✅ 4. 主训练入口
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
        callbacks={'on_fit_epoch_end': on_fit_epoch_end},  # 记录 loss
    )

    if os.path.exists('runs/train/exp/weights/best.pt'):
        log_static_model_info(model)
