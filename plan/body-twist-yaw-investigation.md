# 体のひねり (body twist) がアバターに反映されない問題 — 調査レポート

## 症状

ユーザー報告: 「姿勢推論に関して、体のひねりが全くアバターに反映されない」。
特に正面〜45°程度のひねりで、アバターの胴体が回転しない。

## 結論

**コードのバグではなく、RTMW3D の per-joint depth (Z) 出力の精度不足が根本原因。**
`compute_body_yaw_3d` の数式・適用パスは正常に動作しているが、両肩の Z 座標差が
極端な姿勢以外でほぼ 0 しか出ないため、`atan2(bz, bx)` の出力が 1° 未満に潰れて
可視化上ほぼ無回転になる。

## 確認手順

1. `diagnose_pose` バイナリに validation 画像を入力。
2. `pose_solver::solve_avatar_pose` 内に一時的な `eprintln!` を仕込み、
   - `preprocess_source` 前後の LeftShoulder/RightShoulder の position
   - `compute_body_yaw_3d` の戻り値
   - Hips への local rotation 適用直前/直後の値、blend 係数
   を出力。
3. `validation_images/gothic_lolita_rotation/anime/` の 8 枚 (000°〜315°) を一通り走らせる。

## 計測結果

| 入力角度 | LS pos (x, y, z) | RS pos (x, y, z) | bx | bz | 算出ヨー |
|---|---|---|---|---|---|
| 000° front | +0.012, +0.474, -0.089 | -0.160, +0.464, +0.002 | 0.172 | -0.091 | (※未計測, 別フレームで再測必要) |
| 045° three-quarter | +0.059, +0.486, **-0.083** | -0.155, +0.486, **-0.087** | 0.214 | **+0.004** | **+0.93°** |
| 090° side profile | +0.054, +0.534, +0.019 | -0.029, +0.531, -0.075 | 0.083 | +0.094 | +48.2° |
| 135° rear three-quarter | -0.130, +0.486, +0.073 | +0.094, +0.496, +0.066 | -0.224 | +0.007 | +178.2° |
| 180° back | -0.166, +0.504, +0.061 | +0.140, +0.520, +0.047 | -0.306 | +0.014 | +177.4° |
| 225° rear three-quarter | -0.153, +0.513, +0.064 | +0.122, +0.529, +0.054 | -0.275 | +0.010 | +177.8° |
| 270° side profile | +0.031, +0.523, +0.012 | -0.017, +0.523, -0.078 | 0.048 | +0.090 | +61.9° |
| 315° three-quarter | +0.179, +0.461, -0.023 | -0.068, +0.461, -0.026 | 0.247 | -0.004 | -0.81° |

### 観察

- **45° / 315° の三クォータ姿勢で両肩の Δz は 0.004 しか出ない** (X方向の Δx は 0.21 前後あるのに対し、ノイズフロアと同程度)。
- **90° / 270° (真横向き) でようやく Δz が 0.09 まで出る** が、本来期待される ~0.21 (= rest 時の Δx) には届かず、ヨーは 48–62° と過小出力。
- **135° / 180° / 225° の背中向きは X の符号反転で正しく拾える** が、これは Z 信号ではなく X 信号によるもの。
- 肘 (LowerArm) の Δz は 45° 姿勢でも 0.189 と十分に大きい (肩は 0.004)。これは「同じ深さ平面に近い 2点 (肩同士) は SimCC のヨー解像度では分離しきれない」という RTMW3D の特性を示唆。

### Hips 適用パスの動作確認 (45° 姿勢)

```
[body_yaw_debug] body_yaw=0.93deg, abs>0.0100=true
[body_yaw_debug] hips_node found at idx 1
[body_yaw_debug] prev_local_rot=[0.06289843, 0.0, 0.0, 0.99801993]
[body_yaw_debug] new_local_rot=[0.06289636, 0.008079872, -0.00050921954, 0.9979872]
[body_yaw_debug] parent_world_rot=[0.0, 0.0, 0.0, 1.0]
[body_yaw_debug] rest_world_rot=[0.06289843, 0.0, 0.0, 0.99801993]
[body_yaw_debug] blend factor=1.0000
[body_yaw_debug] result local_transforms[hips].rotation=[0.06289636, 0.008079872, -0.00050921954, 0.9979872]
```

`new_local_rot` の Y 成分 (0.008) が body_yaw=0.93° に対応する微小値。
**コードパスは正しく動いており、入力の body_yaw 自体が小さい**。

## 根本原因

RTMW3D は SimCC の X/Y/Z bin を独立に argmax する 3D ポーズモデル。
- X bin: 576, Y bin: 768, Z bin: 576。
- nz は hip-centred で出力され、`sz = -(j.nz - origin_nz) * 2.0` でソース座標に変換。
- 解像度上は 1bin = 0.0035 だが、**「胴体の左右差程度の小さな depth 差」を学習データから安定して回帰させられていない** (おそらく訓練分布で 3/4 view の比率が低い、もしくは座標系の自由度が高すぎる)。
- 結果として、両肩のような近接点では Z 差が ほぼ 0 〜数 bin のノイズ程度しか出ない。

`src/avatar/pose_solver.rs:731-747` の `compute_body_yaw_3d` は数学的には正しいが、
入力信号 (両肩 Z 差) 自体が情報を持っていない以上、この実装単独では 45° 程度のひねりを
拾えない。

## 改善案

優先度高い順:

### A. **肩 + 腰の Δz を平均**してロバスト化

`LeftShoulder/RightShoulder` だけでなく `Left/RightUpperLeg` も併用する。
独立ノイズと仮定すれば S/N が √2 程度改善。実装は 10 行程度。

骨格腰側のサンプル測定値 (45° 姿勢):
- LeftUpperLeg: (+0.046, -0.004, **-0.007**)
- RightUpperLeg: (-0.046, +0.004, **+0.007**)
- Δz = -0.014 (肩の 0.004 より大きく、~9° のヨーを生む)

肩と腰の重み付き平均 (信頼度で重み) で `bz_combined`, `bx_combined` を作って
`atan2(bz_combined, bx_combined)` する形が自然。

### B. **トルソ4点 (LS, RS, LH, RH) に平面フィット**

特異値分解で平面の法線を求め、その XZ 投影をヨーに使う。
A よりロバストだが SVD の実装/依存が増える。
A の重み付き平均で十分な改善が得られるならわざわざ採用しない。

### C. **肘 Δz をフォールバックに使う**

肘の Δz は 45° 姿勢でも 0.189 と十分大きい。
ただし「腕を体の側に下ろしている」姿勢でしか有効でなく、腕を上げている状態だと
体のひねりとは関係ないシグナルになる。`UpperArm` の方向と組み合わせて、
「上腕がほぼ垂直に近いとき限定で肘 Δz を使う」など条件付きで採用する設計が必要。

### D. **別 backend への切替**

MediaPipe Pose Landmarker (BlazePose GHUM 3D) や Sapiens に置換。
工数大、互換性の影響範囲も大きい。最終手段。

## 推奨

**まず A (肩 + 腰の Δz の信頼度重み付け平均) を入れる。**
A だけで実用上問題なくなるか検証。不足するようなら C を信頼度ゲート付きで追加。
B/D はその後の選択肢。

## 進捗

- **Option A 実装済 (2026-05-01)**。`compute_body_yaw_3d` を肩ペアと腰
  ペアの信頼度重み付き平均で合成するよう変更。`pose_solver.rs` 内に
  `body_yaw_tests` モジュールを追加し、(1) 上の 45° 計測値を使った
  「肩のみの ~1° よりは大きく出る」回帰テスト、(2) 肩のみ frame での
  シングルペアフォールバック、(3) 全 None 入力、(4) 肩低信頼度+腰高信頼度の
  4 ケースを green。実機検証 (`gothic_lolita_rotation/photorealistic/`
  8 枚) は別途。
- B / C / D は未着手。A の効果を実機で見てから判断。

## 影響ファイル

- `src/avatar/pose_solver.rs` `compute_body_yaw_3d` — A 適用済 (重み付き
  平均で `bx`/`bz` を合成、片方ペアのみ confident でもフォールバック)。
- dead-zone 閾値 (現 0.05) は据え置き。実測でなお過剰トリガするようなら
  再調整。

## 後始末

- 一時的な `eprintln!` 群はすべて削除済み (`src/avatar/pose_solver.rs` を rest 状態に戻し済)。
- 計測ダンプは `C:\Users\kojiro\AppData\Local\Temp\twist_*` に残置 (再検証時に参照可)。

## 一次資料

- `pose_solver.rs` `compute_body_yaw_3d` (現在は肩+腰ペア合成版)
- `pose_solver.rs` Hips 適用パス (`solve_avatar_pose` 内)
- `tracking/rtmw3d/skeleton.rs` ソース座標変換 (`build_source_skeleton`)
- `tracking/rtmw3d/consts.rs` `COCO_BODY` (selfie mirror マッピング)
- `validation_images/gothic_lolita_rotation/photorealistic/` 計測対象画像
