import os
import time
import warnings
import wandb
from ultralytics import RTDETR  # 你改版的 RT-DETR 必须继承这个结构或兼容它

# ✅ 设置 WandB API key（避免每次粘贴）
os.environ["WANDB_API_KEY"] = "f8cb8b13b090d70eb2b9b5ee36da161979b90a95"

warnings.filterwarnings('ignore')

# ✅ 初始化 wandb 项目
wandb.init(
    project='rtdetr-project',
    name=time.strftime("rtdetr_%Y%m%d_%H%M%S")
)

if __name__ == '__main__':
    # ✅ 初始化模型（路径按你实际的 yaml 配置来）
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')

    # ✅ 启动训练
    results = model.train(
        data='/root/dataset/dataset_visdrone/data.yaml',
        imgsz=640,
        epochs=300,
        batch=4,
        workers=4,
        project='runs/train',
        name=wandb.run.name,  # 和 wandb 名保持一致
    )

    # ✅ 每轮记录 loss，如果 results 是完整列表（每个 epoch 的指标），就记录所有 epoch
    try:
        if isinstance(results, list):  # 改版 RT-DETR 有可能返回多个 epoch 的结果
            for i, epoch_metrics in enumerate(results):
                wandb.log({
                    'epoch': i,
                    'loss_cls': epoch_metrics.get('loss_cls', 0),
                    'loss_box': epoch_metrics.get('loss_bbox', 0),
                    'loss_obj': epoch_metrics.get('loss_obj', 0),
                    'loss_total': sum([
                        epoch_metrics.get('loss_cls', 0),
                        epoch_metrics.get('loss_bbox', 0),
                        epoch_metrics.get('loss_obj', 0)
                    ])
                })
        elif isinstance(results, dict):
            # 单次记录（最后一轮）
            wandb.log({
                'loss_cls': results.get('loss_cls', 0),
                'loss_box': results.get('loss_bbox', 0),
                'loss_obj': results.get('loss_obj', 0),
                'loss_total': sum([
                    results.get('loss_cls', 0),
                    results.get('loss_bbox', 0),
                    results.get('loss_obj', 0)
                ])
            })
        else:
            print("⚠️ 无法解析训练返回结果，无法记录 loss")
    except Exception as e:
        print(f"⚠️ WandB 日志记录失败: {e}")
