# Prerequisites

**Please ensure you have prepared the environment and the nuScenes dataset.**

# Train and Test

Train MGMap with 2 GPUs 
```
./tools/dist_train.sh ./projects/configs/mgmap/mgmap_cam_r50_30e.py 2
```

Evaluate GeMap with 2 GPUs
```
./tools/dist_test_map.sh projects/configs/mgmap/mgmap_cam_r50_30e.py ./path/to/ckpts.pth 2
```

