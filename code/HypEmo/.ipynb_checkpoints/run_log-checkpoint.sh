============================================== 结果复现, 跑5组 random_seed 求平均 =====================================
GE
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 1234
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 2345
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 3456
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 4567
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 5678

ED
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 1234
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 2345
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 3456
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 4567
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 5678


=================================  改gamma, alpha为可训练参数 =========================
============================================= 初始都为1.0 ==================================================

GE
python train.py --dataset go_emotion --batch_size 16 --seed 1234
python train.py --dataset go_emotion --batch_size 16 --seed 2345
python train.py --dataset go_emotion --batch_size 16 --seed 3456
python train.py --dataset go_emotion --batch_size 16 --seed 4567
python train.py --dataset go_emotion --batch_size 16 --seed 5678

ED
python train.py --dataset ED --batch_size 16 --seed 1234
python train.py --dataset ED --batch_size 16 --seed 2345
python train.py --dataset ED --batch_size 16 --seed 3456
python train.py --dataset ED --batch_size 16 --seed 4567
python train.py --dataset ED --batch_size 16 --seed 5678

============================================= 初始都为0.5 ==================================================
GE
python train.py --dataset go_emotion --batch_size 16 --seed 1234
python train.py --dataset go_emotion --batch_size 16 --seed 2345
python train.py --dataset go_emotion --batch_size 16 --seed 3456
python train.py --dataset go_emotion --batch_size 16 --seed 4567
python train.py --dataset go_emotion --batch_size 16 --seed 5678

ED
python train.py --dataset ED --batch_size 16 --seed 1234
python train.py --dataset ED --batch_size 16 --seed 2345
python train.py --dataset ED --batch_size 16 --seed 3456
python train.py --dataset ED --batch_size 16 --seed 4567
python train.py --dataset ED --batch_size 16 --seed 5678


========================================================  cross attention ============================================
GE
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 1234
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 2345
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 3456
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 4567
python train.py --dataset go_emotion --alpha 0.9 --gamma 0.1 --batch_size 16 --seed 5678
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 1234
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 2345
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 3456
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 4567
python train.py --dataset ED --alpha 1.0 --gamma 0.25 --batch_size 16 --seed 5678
