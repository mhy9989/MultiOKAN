from easydict import EasyDict as edict

def default_parser():
    default_values = {
      "setup_config":{
        "seed": 9989,
        "diff_seed": False,
        "per_device_train_batch_size": 128,
        "per_device_valid_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "num_workers": 12,
        "method": "l-deepokan",
        "max_epoch": 1000,
        "lossfun": "MSE",
        "load_from": False,
        "if_continue": False,
        "regularization": 0.0,
        "if_display_method_info": True,
        "mem_log": False,
        "empty_cache": False,
        "metrics":["MSE", "RMSE", "MAE", "MRE", "SSIM"],
        "fps": True,
        "drop_last": False,
        "val_metrics": True
      },
      "data_config": {
        "org_path": "../dataset.npy",
        "mesh_path": "../mesh.npy",
        "data_num": 1000,
        "data_width": 128,
        "data_height": 128,
        "data_mean": [[0.5]],
        "data_std": [[0.1]],
        "data_max": [[0.1]],
        "data_min": [[0]],
        "data_type": ["solid"],
        "data_select":[0],
        "data_scaler": ["MinMax"],
        "total_after": 100,
        "data_after_num": 10,
        "dt": 0.01,
        "train_seed":9989,
        "train_ratio": 0.9,
        "valid_ratio": 0.0,
        "test_ratio": 0.1,
        "t_norm": "Standard"
      },
      "optim_config": {
        "optim": "Adamw",
        "lr": 2e-4,
        "filter_bias_and_bn": False,
        "log_step": 1,
        "opt_eps": "",
        "opt_betas": "",
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "early_stop_epoch": -1
      },
      "sched_config": {
        "sched": "onecycle",
        "min_lr": 1e-6,
        "warmup_lr": 1e-5,
        "warmup_epoch": 0,
        "decay_rate": 0.1,
        "decay_epoch": 100,
        "lr_k_decay": 1.0,
        "final_div_factor": 1e4
      },
      "model_config": {
        "kan_type": "gram",
        "model_type": "AE_fusion",
        "AE_layers":[1008, 144],
        "latent_dim":144,
        "actfun":"silu",
        "norm": True,
        "kan_config":{
          "degrees": 3
        }
      },
      "ds_config":{
        "offload": False,
        "zero_stage": 0,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 2.0
      }}
    return edict(default_values)
