# GM V24.3 Native KNN & Neural Net (MLP)

ATH ref **0.95440** · elapsed 1368s

## Hypotheses

1. **KNN:** local neighborhoods in scaled+PCA space carry fraud signal not captured by global linear/logit blends.
2. **MLP:** depth-2 ReLU net on the same FE learns interactions complementary to trees / ATH dual-base.

## OOF results

| Model | OOF AUC | Notes |
|---|---:|---|
| KNN k=101 | 0.914905 | PCA-24, 50k prototypes, distance weights |
| KNN k=201 | 0.916217 | |
| **KNN best (k=201)** | **0.916217** | canary 120k/PCA32≈0.91 |
| MLP (256,128) | 0.938308 | folds 0.9376, 0.9395, 0.9425, 0.9395, 0.9332 |
| MLP (512,) | 0.939421 | folds 0.9367, 0.9402, 0.9447, 0.9405, 0.9354 |
| **Selected MLP (`mlp_512`)** | **0.939421** | |
| KNN+MLP(+HGB) blend (0.1, 0.2, 0.7) | **0.944967** | |

## Nested gates

| model | vs ATH-orth | vs gb+8% | ρ(test,ATH)e-4 |
|---|---|---|---:|
| `knn` | 0/5 t=-1.00 m=-0.00 | 1/5 t=-2.53 m=-13.62 | 2333.2 |
| `mlp` | 1/5 t=1.00 m=+0.00 | 3/5 t=0.15 m=+0.37 | 990.6 |
| `blend` | 0/5 t=0.00 m=+0.00 | 5/5 t=1.70 m=+5.02 | 621.3 |

## Live submits

| tag | Public | status |
|---|---:|---|
| `ATH_x_knn_mlp_0.08` | **None** | quota_or_error |

- Best live: **None**

## Expert takeaway

- Solo KNN/MLP will not beat 0.95440; expect OOF in the mid-0.93–0.94s unless FE is upgraded.
- Value is **diversity into ATH** at small w if `gb+8%` nested gate is green.
- Next reset: submit staged `ATH_x_mlp_*` / `ATH_x_knn_mlp_*` before pure natives.

## Log
```
[    0.0s] === V24.3 Native KNN + MLP (expert DS) ===
[    0.3s] Building feature matrix…
[   12.4s] X=(699960, 62) fraud=0.01670
[   12.5s] Training KNN (scaled → PCA-24 → 50k prototypes → distance-weighted)…
[  374.7s]   KNN fold0 k101=0.90874 k201=0.91024 pca_var=0.621 proto=50000
[  554.3s]   KNN fold1 k101=0.91718 k201=0.91965 pca_var=0.621 proto=50000
[  733.7s]   KNN fold2 k101=0.92171 k201=0.92318 pca_var=0.620 proto=50000
[  916.4s]   KNN fold3 k101=0.91683 k201=0.91691 pca_var=0.620 proto=50000
[ 1089.0s]   KNN fold4 k101=0.91061 k201=0.91184 pca_var=0.621 proto=50000
[ 1089.3s] KNN OOF AUCs={101: 0.9149050289723454, 201: 0.9162171294631598} best_k=201
[ 1089.3s] Training MLPClassifier (256,128) with early stopping…
[ 1122.6s]   MLP fold0 AUC=0.937588 n_iter=13
[ 1159.9s]   MLP fold1 AUC=0.939506 n_iter=14
[ 1179.2s]   MLP fold2 AUC=0.942456 n_iter=11
[ 1196.0s]   MLP fold3 AUC=0.939478 n_iter=10
[ 1214.0s]   MLP fold4 AUC=0.933211 n_iter=11
[ 1214.2s] MLP OOF=0.938308 mean_fold=0.938448
[ 1214.2s] Training MLP-wide (512,) …
[ 1236.1s]   MLP2 fold0 AUC=0.936716 n_iter=10
[ 1265.2s]   MLP2 fold1 AUC=0.940191 n_iter=14
[ 1296.2s]   MLP2 fold2 AUC=0.944659 n_iter=15
[ 1317.9s]   MLP2 fold3 AUC=0.940484 n_iter=10
[ 1335.8s]   MLP2 fold4 AUC=0.935391 n_iter=8
[ 1335.9s] MLP2 OOF=0.939421 mean_fold=0.939488
[ 1336.3s] loaded HGB OOF=0.944568 for optional mix
[ 1346.1s] KNN+MLP(+HGB?) blend OOF=0.944967 w={'knn': 0.1, 'mlp': 0.2, 'hgb': 0.7}
[ 1349.8s] ATH bank surrogate AUC=0.951428
[ 1351.5s] GATE knn: ATH-orth 0/5 t=-1.00 m=-0.00 | gb+8% 1/5 t=-2.53 m=-13.62 | solo=0.91622 ρ(ATH)=2333.2e-4
[ 1353.0s] GATE mlp: ATH-orth 1/5 t=1.00 m=+0.00 | gb+8% 3/5 t=0.15 m=+0.37 | solo=0.93942 ρ(ATH)=990.6e-4
[ 1354.6s] GATE blend: ATH-orth 0/5 t=0.00 m=+0.00 | gb+8% 5/5 t=1.70 m=+5.02 | solo=0.94497 ρ(ATH)=621.3e-4
[ 1355.7s] SAVED /tmp/s6e11_v16/out/v24_3_native_knn_mlp.npz
[ 1357.1s] STAGED sub_v24_3_native_knn.csv ρ(ATH)=2333.181e-4
[ 1358.0s] STAGED sub_v24_3_native_mlp.csv ρ(ATH)=990.605e-4
[ 1358.9s] STAGED sub_v24_3_native_knn_mlp_blend.csv ρ(ATH)=621.263e-4
[ 1359.7s] STAGED sub_v24_3_ATH_x_knn_0.05.csv ρ(ATH)=5.906e-4
[ 1360.5s] STAGED sub_v24_3_ATH_x_mlp_0.05.csv ρ(ATH)=2.412e-4
[ 1361.9s] STAGED sub_v24_3_ATH_x_knn_mlp_0.08.csv ρ(ATH)=3.886e-4
[ 1363.5s] STAGED sub_v24_3_ATH_x_mlp_0.10.csv ρ(ATH)=9.724e-4
[ 1363.7s] SUBMIT v24.3 ATH_x_knn_mlp_0.08 | ρ(ATH)=3.886e-4 | KNN201=0.9162 mlp_512=0.9394 blend=0.9450
[ 1368.4s] 
[ 1368.4s] 0%|          | 0.00/7.52M [00:00<?, ?B/s]
  0%|          | 16.0k/7.52M [00:00<00:58, 134kB/s]
  9%|▊         | 656k/7.52M [00:00<00:05, 1.25MB/s]
 10%|█         | 784k/7.52M [00:00<00:06, 1.09MB/s]
 13%|█▎        | 1.00M/7.52M [00:00<00:06, 1.11MB/s]
[ 1368.4s] DONE ATH_x_knn_mlp_0.08 pub=None status=quota_or_error
[ 1368.4s] STOP quota — remaining staged
```

