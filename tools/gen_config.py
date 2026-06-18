import json
import os
from pathlib import Path

# === 1. 定义输出目录 (新名字：aug-hard) ===
output_dir = Path("outputs/train/mact-pusht-final-v4-aug-hard")
output_dir.mkdir(parents=True, exist_ok=True)
config_path = output_dir / "train_config.json"

# === 2. 魔鬼训练配置 (Hard Augmentation) ===
config_data = {
    "dataset": {
        "repo_id": "lerobot/pusht_image",
        "image_transforms": {
            "enable": True,
            "max_num_transforms": 3,
            "random_order": False,
            "tfs": {
                "affine": {
                    "type": "RandomAffine",
                    "weight": 1.0,
                    "kwargs": {
                        "degrees": [-10.0, 10.0],
                        "translate": [0.15, 0.15],
                        "scale": [0.9, 1.1]
                    }
                },
                "brightness": {
                    "type": "ColorJitter",
                    "weight": 1.0,
                    "kwargs": {"brightness": [0.8, 1.2]}
                },
                "contrast": {
                    "type": "ColorJitter",
                    "weight": 1.0,
                    "kwargs": {"contrast": [0.8, 1.2]}
                },
                "blur": {
                    "type": "GaussianBlur",
                    "weight": 0.5,
                    "kwargs": {"kernel_size": 3}
                }
            }
        },
        "use_imagenet_stats": True,
        "video_backend": "pyav"
    },
    "env": {
        "type": "pusht",
        "task": "PushT-v0"
    },
    "policy": {
        "type": "mact",
        "n_obs_steps": 2,
        "chunk_size": 50,
        "n_action_steps": 10,
        "dropout": 0.15,
        "use_memory": True,
        "optimizer_lr": 1e-4,
        "optimizer_lr_backbone": 1e-5,
        "use_amp": True,
        "vision_backbone": "resnet18", 
        "repo_id": "mact-pusht-final-v4-aug-hard"
    },
    "optimizer": {
        "lr": 1e-4,
        "weight_decay": 5e-3
    },
    "batch_size": 64,
    "steps": 150000, 
    "eval_freq": 10000,
    "save_freq": 20000,
    "save_checkpoint": True,
    "num_workers": 8,
    "seed": 3000,
    "output_dir": str(output_dir),
    "job_name": "pusht_mact_aug",
    
    # ⬇️⬇️⬇️ 关键修改：第一次运行必须是 False ⬇️⬇️⬇️
    "resume": False 
}

# === 3. 写入文件 ===
with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)

print(f"✅ 最终修正版配置已生成: {config_path}")
