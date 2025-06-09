import warnings, os, time
os.environ['WANDB_API_KEY'] = 'f8cb8b13b090d70eb2b9b5ee36da161979b90a95'  # ⚠️ 仅用于测试时使用环境变量，不推荐硬编码

# 启用 wandb 在线模式（可选）
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_INIT_TIMEOUT"] = "180"

warnings.filterwarnings('ignore')
from ultralytics import RTDETR
import wandb  # ← 加上这一行导入 wandb

if __name__ == '__main__':
    run = wandb.init(project="redet_backbone", name=time.strftime("%Y-%m-%d_%H-%M-%S"))  # ← 初始化 wandb 项目
    model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-r18.yaml')

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
