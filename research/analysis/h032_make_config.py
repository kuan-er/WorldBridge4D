"""Generate the minimal reproducible H032 protocol diff from the196k parent."""
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
def make_config():
    cfg=yaml.safe_load((ROOT/'configs/h031_k512_dr512_full_k9_resume191000_to200000.yaml').read_text())
    cfg['full_mode_unfreeze_resume']=False
    cfg['camera_supervision']=dict(dim=256,num_heads=8,memory_grid=16,seed=424243,
        learning_rate=1e-4,warmup_steps=500,phase_start_step=196000,skip_zero_weight_cycle=True,
        loss_weights=dict(diagonal_xyz=0.5,offdiagonal_xyz=0.5,ray=0.1,front=0.01,
                          pose_rotation=0.1,pose_translation=0.1,fov=0.1))
    cfg['expected_non_wan_parameters']=194597133+1414409
    cfg['finetune_expected_global_step']=196000
    cfg['finetune_expected_clips_seen']=dict(kubric=600664,pointodyssey=453120,dynamic_replica=514216)
    cfg['max_steps']=210000
    cfg['lr_restart']['end_step']=210000
    cfg['lr_restart']['group_learning_rates']['camera_head']=1e-4
    cfg['checkpoint_steps']=list(range(197000,210001,1000))
    cfg['tracking']['resume']='allow'
    cfg['tracking']['group']='h032-source-conditioned-camera-ray-196000-to210000'
    cfg['tracking']['tags']=['h032','source-conditioned-camera','gt-ray','shared-fov','b1-a4-k9','native512-full','new-branch']
    return cfg
if __name__=='__main__':
    (ROOT/'configs/h032_camera_ray_196000_to210000.yaml').write_text(yaml.safe_dump(make_config(),sort_keys=False))
