# スキニング後の多層衣装をGPUで補正する：階層アンカーによるクリアランス制約の実装と評価
## 〜ボーン放射距離ヒューリスティックによる内外順序推定と、Vulkanコンピュートスキニングによる局所突き破り軽減〜

> **著者**: VulVATAR Graphics & Tracking Engineering Team  
> **タグ**: `Vulkan` `GPU Compute` `Computer Graphics` `Skeletal Animation` `Clothing Simulation`  

---

## 1. はじめに：リアルタイムアバターにおける「多層衣装クリッピング」

VTuber、メタバース、VRChat、ゲーム開発など、リアルタイム3Dキャラクター描画において頻繁に直面する課題の一つが**「衣装の突き破り（クリッピング・貫通）」**です。

特に問題となりやすいのが、**「シャツの上にブレザーを羽織る」「下着や素体の上にタイトな衣服を着る」といった多層衣装（Layered Clothing）** です。キャラクターが腕を深く曲げたり関節を大きく動かした際、内側の服が外側の服の生地を局所的に突き破って露出してしまう現象が生じます。

### 従来のアプローチと実用上のトレードオフ

この問題に対して、これまで様々なアプローチが試みられてきました：

| 手法 | メリット | トレードオフ・制約 |
|---|---|---|
| **カプセルコライダーの配置** | 物理エンジン（Dynamic Bone, PhysBone等）と親和性が高い | 薄い布層間（数ミリ〜1センチ程度）ではトンネリング（すり抜け）や反発の振動（ジッター）が起きやすい。アバターごとに手動調整が必要。 |
| **不可視メッシュの削除・マスク** | 隠した部分の貫通が画面に現れない | 衣装の着脱や袖口・襟元から覗くメッシュを残す必要がある場合、動的マスクテクスチャや専用シェーダー等の仕組みが必要。 |
| **ボーンウェイトの同期・転写** | 共通ボーンに対する変形傾向が揃う | メッシュ固有のシワや布の厚み表現が制約される場合がある。また、服ごとに独立した二次揺れボーン（胸揺れ・裾揺れ等）が存在する場合、その物理挙動まで完全に同期させることは難しい。 |

本稿では、手動コライダー調整を回避しつつ、スキニング後のGPUパイプラインにおいて内側メッシュを参照して外側メッシュの位置を押し出す**「階層アンカーによるクリアランス補正」** の実装例を報告します。

具体的には、
1. レストポーズにおけるメッシュ間の**ボーン放射距離（Radial Distance）** を用いた内外順序の推定ヒューリスティック
2. 空間距離とボーンウェイト類似度を組み合わせたアンカー対応点探索
3. Vulkan コンピュートスキニングパイプラインにおける **Kahn のアルゴリズムを用いたトポロジカルディスパッチと異常依存時の安全フォールバック**
4. GPU コンピュートシェーダーによる一方向のクリアランス押し出し補正

の設計と実装、および商用アバター（Yumeka）でのテスト結果と本手法が持つ工学的な制約・限界について客観的に解説します。

---

## 2. 関節屈曲時における布地交差の要因

### Yumekaモデルにおける実測例

商用VTuberアバター「Yumeka」の上半身メッシュ構成を調査したところ、以下のレイヤーが存在していました：

- **素体メッシュ** (`Circle.053`)
- **シャツメッシュ** (`Circle.057`, 17,844 頂点)
- **ブレザーメッシュ** (`Circle.051`, 17,172 頂点)

レストポーズ（Tポーズ）における前腕ボーン軸（Yumekaモデルのノード名: `LowerArm_L`、Humanoid共通呼称: `LeftForeArm`）からの放射距離中央値を計測すると：
- シャツの肘ボーン距離中央値: **54.8 mm**
- ブレザーの肘ボーン距離中央値: **62.3 mm**
- （参考：中央値の差は **約 7.5 mm**）

### ボーンウェイトの差異による交差リスク

線形ブレンドスキニング（Linear Blend Skinning: LBS）において、変形後頂点位置 $\mathbf{v}'$ はボーン行列 $\mathbf{M}_j$ とウェイト $w_j$ の加重平均で決まります：

$$\mathbf{v}' = \sum_{j} w_j \mathbf{M}_j \mathbf{v}_{\text{rest}}$$

肘屈曲部にある近接頂点ペアのウェイト配分を調べると、以下のような差異が見られました：

```
Outer（ブレザー頂点例）:
  - UpperArm:         0.69
  - LowerArm:         0.31

Inner（シャツ頂点例）:
  - UpperArm:         0.42
  - LowerArm:         0.58
```

前腕ボーンが90度回転した際、前腕ウェイトの比率が大きいシャツ頂点（0.58）は回転運動に強く追従する一方、ブレザー頂点（0.31）は上腕寄りに留まります。初期クリアランスの厚みや動作範囲によっては、この変形ベクトルの違いによって内側メッシュが外側メッシュの軌道を追い越し、局所的な交差（クリッピング）が発生しやすくなります。

---

## 3. 先行研究との位置づけ

衣服の多層貫通・交差解消に関する代表的な先行研究として、**Buffetらの *Implicit Untangling: A Robust Solution for Modeling Layered Clothing* (ACM TOG 2019)** が知られています。

- **Buffetら (2019) のアプローチ**: ユーザーが指定した層順序と厚みなどに基づいて陰関数表現（Implicit Surfaces）の合成演算子を定義し、その結果へ衣服メッシュを投影することで多層交差を解消する。
- **本手法のアプローチ**: 層順序をメッシュの幾何学的ボーン放射距離からヒューリスティックに自動推定し、リアルタイム描画パイプラインの頂点スキニングステージ直後で固定アンカーによる軽量な点-接平面押し出しを行う工学的な軽量解法。

---

## 4. 手法の実装

### 4.1 ボーン放射距離ヒューリスティックによる内外順序の推定

衣服が「どちらが内側で、どちらが外側か」を判定する際、近接頂点間の法線ベクトル $\mathbf{n}$ と変位ベクトル $\mathbf{d} = \mathbf{p}_B - \mathbf{p}_A$ の内積を見る手法が考えられます。しかし、衣服メッシュにはシワの折り返しや袖口・裏地のポリゴンが含まれており、法線が局所的に反転している場合があるため、法線のみに依存すると誤判定のリスクがあります。

そこで、局所的に入れ子になっている衣服メッシュに対し、**骨格軸セグメント $L = [\mathbf{b}_{\text{start}}, \mathbf{b}_{\text{end}}]$ への直交射影距離（放射距離 $r(\mathbf{p})$）** を補助指標として利用します：

$$r(\mathbf{p}) = \min_{t \in [0, 1]} \|\mathbf{p} - ((1 - t)\mathbf{b}_{\text{start}} + t\mathbf{b}_{\text{end}})\|$$

メッシュ $A$ の近傍点 $\mathbf{p}_A$ とメッシュ $B$ の近傍点 $\mathbf{p}_B$ における放射距離差：

$$\Delta r_{BA} = r(\mathbf{p}_B) - r(\mathbf{p}_A)$$

サンプリングした近接点群において、正値をとる割合が多数（本実装では 90% 以上）を占める場合、「メッシュ $B$ はメッシュ $A$ の外側にある」と推定します。

> [!WARNING]
> **本ヒューリスティックの適用限界と前提条件**
> - **しわや凹凸による局所逆転**: 内側メッシュのしわの山と外側メッシュの平坦部を比較すると、衣服全体として交差していなくても局所的に距離の大小が逆転し得ます。
> - **共有ボーンの必要性**: 比較対象となる両メッシュが共通の骨格セグメントから十分な影響を受けている必要があります。
> - **単一親IDの限界**: 現在のデータ構造（`body_primitive_id: Option<PrimitiveId>`）では、1つの衣服プリミティブが複数の異なる親プリミティブにまたがって接触する複雑な衣装は表現できません。

---

### 4.2 ボーンウェイト類似度を加味したアンカー探索

外側メッシュの各頂点 $\mathbf{p}_o$ に対し、内側メッシュの基準頂点（アンカー）を対応付けます。
レストポーズでのユークリッド距離のみで最近傍を求めると、関節屈曲時に異なる運動をする頂点とペアリングされてしまうため、ボーンウェイトの類似度をペナルティとして目的関数に加えます。

#### コサイン類似度 $S_{\text{bone}}(u, v)$ の定義
ボーンIDを共通の座標軸とする疎なウェイトベクトル（各頂点の非ゼロ要素は最大4個） $\mathbf{W}(u), \mathbf{W}(v)$ に対し、正規化されたコサイン類似度を算出します：

$$S_{\text{bone}}(u, v) = \frac{\sum_{k \in J(u) \cap J(v)} W_k(u) W_k(v)}{\sqrt{\sum_k W_k(u)^2} \sqrt{\sum_k W_k(v)^2}}$$

（※分母が微小値以下の場合は $0$ を返します）。

#### 最適化目的関数
アウター頂点 $\mathbf{p}_o$ に対するアンカー候補 $i^*$ を以下のように選択します：

$$i^* = \arg\min_{i \in \text{Candidates}} \left( \|\mathbf{p}_o - \mathbf{p}_i\|^2 + \lambda \cdot (1 - S_{\text{bone}}(o, i)) \right)$$

ここで $\lambda = 0.0003 \, [\text{m}^2]$ です。また、$S_{\text{bone}} < 0.2$ の候補は探索から除外します。

> [!NOTE]
> **対応関係の性質**
> この $\arg\min$ 探索は各アウター頂点に対して独立に行われるため、インナー頂点に対しては**多対一（Multiple-to-One）** の対応付けとなります。また、ウェイト類似度が高くても、初期位置が異なる頂点同士が同一の軌道を描くわけではありません。

---

### 4.3 Vulkan レンダラーにおけるトポロジカル・ディスパッチと安全フォールバック

衣服プリミティブ間の依存関係（例: 素体 $\to$ シャツ $\to$ ブレザー）が構築された場合、GPU 上で親サーフェスの変形後頂点バッファを先に生成し、それを子サーフェスのコンピュートスキニングで読み込む必要があります。

#### Kahn のアルゴリズムと異常依存時のフォールバック
単純な `sort_by` 比較関数は推移律を満たさず全順序にならないため、**Kahn のアルゴリズム（入次数カウントとキューによる DAG トポロジカルソート）** を採用しています。
さらに、実行時安全性を担保するため、以下の異常が検出されたノードは親依存関係（`validated_parent_ids`）から完全に除外されます：
- **自己参照**（`parent_id == primitive_id`）
- **存在しない親 ID**
- **重複するプリミティブ ID を持つノード自身**
- **重複する親 ID を参照している子ノード**（親が一意に定まらないため）
- **循環依存**（Kahn 法のキューで未解決のまま残ったノード群）

```rust
// Topological dependency ordering for hierarchical surface clearances:
// If primitive B specifies body_primitive_id = Some(A), A must be dispatched before B.
// Using Kahn's algorithm with cycle detection and safe runtime fallback.
let n = instance.mesh_instances.len();
let mut in_degree = vec![0usize; n];
let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

// 重複 ID の検出
let mut prim_id_to_idx = HashMap::new();
let mut duplicate_ids = std::collections::HashSet::new();
for (idx, mi) in instance.mesh_instances.iter().enumerate() {
    if prim_id_to_idx.insert(mi.primitive_id, idx).is_some() {
        duplicate_ids.insert(mi.primitive_id);
        warn!("render: duplicate primitive_id {:?} in mesh instances", mi.primitive_id);
    }
}

let mut validated_parent_ids: HashMap<PrimitiveId, PrimitiveId> = HashMap::new();

for (idx, mi) in instance.mesh_instances.iter().enumerate() {
    if duplicate_ids.contains(&mi.primitive_id) { continue; }

    if let Some(parent_id) = mi.primitive_data.as_ref().and_then(|p| p.body_primitive_id) {
        // 重複親を参照する子は一意に親を解決できないためプルーニング
        if duplicate_ids.contains(&parent_id) {
            warn!("render: parent {:?} is ambiguous (duplicates); pruning child {:?}", parent_id, mi.primitive_id);
            continue;
        }

        if let Some(&parent_idx) = prim_id_to_idx.get(&parent_id) {
            if parent_idx != idx {
                adj[parent_idx].push(idx);
                in_degree[idx] += 1;
                validated_parent_ids.insert(mi.primitive_id, parent_id);
            } else {
                warn!("render: self-referencing body_primitive_id {:?} pruned", parent_id);
            }
        } else {
            warn!("render: missing parent {:?}; fallback to unconstrained skinning", parent_id);
        }
    }
}

let mut queue: std::collections::VecDeque<usize> = in_degree
    .iter()
    .enumerate()
    .filter_map(|(idx, &deg)| if deg == 0 { Some(idx) } else { None })
    .collect();

let mut ordered_mesh_instances = Vec::with_capacity(n);
let mut visited = vec![false; n];

while let Some(u) = queue.pop_front() {
    visited[u] = true;
    ordered_mesh_instances.push(&instance.mesh_instances[u]);
    for &v in &adj[u] {
        in_degree[v] -= 1;
        if in_degree[v] == 0 { queue.push_back(v); }
    }
}

// 循環依存ノードのフォールバック：親参照を削除して通常スキニングへ戻す
if ordered_mesh_instances.len() < n {
    warn!("render: cycle detected; disabling clearance on cyclic nodes");
    for (idx, mi) in instance.mesh_instances.iter().enumerate() {
        if !visited[idx] {
            validated_parent_ids.remove(&mi.primitive_id);
            ordered_mesh_instances.push(mi);
        }
    }
}
```

#### フォールバックの実行時経路
無効化された親マッピングは、ディスパッチの並び順だけでなく、以下の全実行経路に直結しています：
1. **親バッファの参照**: `validated_parent_ids` に存在しないプリミティブは `prim_parent_vbo = None` となり、親バッファは取得されません。
2. **シェーダー制御 UBO**: `guard.has_skin_anchors = if prim_parent_vbo.is_some() { 1 } else { 0 }` により、制御フラグが `0` となります。
3. **GPU Compute Shader**: シェーダー内のクリアランス補正分岐（`if (push.has_skin_anchors != 0)`）が完全にスキップされ、**通常のスキニング（unconstrained skinning）へ安全に復帰**します。

#### GPU メモリの同期
同一コマンドバッファ内の追跡対象リソースへのアクセスについて、本パイプラインでは Vulkano 0.35 の `AutoCommandBufferBuilder` の自動同期を利用しています（先行ディスパッチの `transformed_vbo` 書き込みと、後続ディスパッチの Binding 7 での読み込みの依存関係追跡）。

---

### 4.4 GPU コンピュートシェーダーによる位置補正

変形後頂点バッファを生成するコンピュートシェーダー（`transform_cs.comp`）において、外側メッシュの頂点位置を親メッシュの接平面に対して押し出します：

```glsl
// GLSL: transform_cs.comp (外側頂点の一方向位置補正)
if (push.has_skin_anchors != 0) {
    SkinAnchor anchor = skin_anchors[gl_GlobalInvocationID.x];
    if (anchor.weight > 0.0) {
        vec3 parent_p = parent_v[anchor.body_vertex_idx].pos;
        vec3 raw_n = parent_v[anchor.body_vertex_idx].norm;
        float n_len = length(raw_n);
        vec3 parent_n = (n_len > 1e-5) ? (raw_n / n_len) : vec3(0.0, 1.0, 0.0);

        vec3 delta = skinned_pos - parent_p;
        float current_clearance = dot(delta, parent_n);

        if (current_clearance < anchor.min_clearance) {
            float push_dist = anchor.min_clearance - current_clearance;
            skinned_pos += parent_n * (push_dist * anchor.weight);
        }
    }
}
```

> [!IMPORTANT]
> **幾何学的補正の性質と制限事項**
> 1. **一方向補正**: 本処理は外側頂点を外側に押し出す「一方向の位置補正（Unilateral Projection）」です（内側頂点を押し戻す処理ではありません）。
> 2. **重み $w < 1$ での未到達**: クリアランス現在値を $d$、必要値を $c$、重みを $w$ とすると補正後距離は $d' = d + w(c - d)$ となり、$w < 1$ の場合は設定クリアランスに届きません。完全なクリアランスを要求する場合は $w = 1.0$ が前提となります。
> 3. **点-接平面拘束の限界**: この補正が保証するのは「選ばれた親頂点の接平面に対する点-平面距離」のみです。三角形同士のメッシュ交差、エッジ交差、補正によって新たに生じる二次的な自己交差は検出・防止できません。
> 4. **法線の再計算未実施**: 補正による形状変化に対する法線の再計算は行っておらず、拘束にはスキニング法線による近似平面を使用しています。

---

## 5. テスト検証と実測結果

### テスト条件と検証指標
- **対象モデル**: 商用アバター Yumeka（FBX）
- **対象部位**:
  - Inner: `Circle.057` (シャツ, 17,844 頂点)
  - Outer: `Circle.051` (ブレザー, 17,172 頂点)
- **テストケース**: [`test_layered_clothing_clearance_e2e`](file:///C:/lib/github/kjranyone/VulVATAR/src/asset/fbx/tests.rs)
- **ポーズ設定**: 左前腕ボーン（`LowerArm_L`）をローカル X 軸に沿って **90度屈曲**
- **検証環境**: CPU シミュレーションテスト（`compute_rest_world_vertices` による LBS 再現計算）

#### 計測指標の定義
本テストにおける計測は、メッシュ全体の三角形交差判定ではなく、**「肘中心から半径 70mm 以内にあるインナー頂点に対し、最も近いアウター頂点のアウター法線方向に対する符号付き距離 $(\mathbf{p}_{\text{inner}} - \mathbf{p}_{\text{outer}}) \cdot \mathbf{n}_{\text{outer}} > 1.0\,\text{mm}$（アウター接平面の外側へ1mm超過で突出）を満たす頂点数」** を評価対象としています（意図する健全な配置はアウター接平面より内側）。

### 実測値

```
======================================================================
  多層衣装クリアランス補正 テスト計測結果 (Yumeka 肘 90度 屈曲)
======================================================================
【アンカー構築状況】
  - Inner 検出:     Circle.057 (シャツ)
  - Outer 検出:     Circle.051 (ブレザー)
  - アンカー確立数: 16,989 / 17,172 頂点 (98.9%)
  - 未バインド数:   183 頂点 (1.1% - 影響半径外または類似度閾値未満)

【肘 90度 屈曲時の点-接平面距離計測】
  - 補正前 (Skinned Baseline):
      法線方向突出が 1mm を超えた頂点数: 75 頂点
      最大法線方向突出 (超過分):        26.38 mm
  - 補正後 (Clearance Projected):
      法線方向突出が 1mm を超えた頂点数: 0 頂点 (該当なし)
      最大突出値:                       未計測 (※1.0mm 以下の微小突出が残存し得る)
======================================================================
```

補正前はウェイト差によりシャツ頂点がブレザーの接平面より最大 26.38 mm 外側に突出していましたが、アンカーに基づく押し出し補正を適用した結果、計測対象領域において突出が 1mm を超える頂点は 0 頂点となりました。

> [!NOTE]
> **ベンチマークに関する留意事項**
> - **閾値境界と微小突出の可能性**: 判定条件が `> 1.0mm` であるため、補正後の「超過頂点数 0 頂点」は 1.0mm を超える突出が検出されなかったことを意味します。本テストでは閾値超過頂点がない場合に最大値集計を行わないため、**1.0mm 以下の微小な突出が残存している可能性は排除しておらず、最大突出値自体は未計測** です。
> - **CPU シミュレーション**: 本テストは CPU 上で LBS 変形とクリアランス補正関数を実行したシミュレーション検証であり、実際の Vulkan レンダリングパスにおける GPU フレームタイムの確定値やジッターを評価するものではありません。
> - **一般化の制限**: 本結果は特定モデル（Yumeka）の肘屈曲ポーズにおける検証結果であり、任意の未知アバターや激しい動的アニメーションに対して破綻ゼロを一般的に証明するものではありません。

---

## 6. まとめと今後の課題

本稿では、多層衣装のスキニング時クリッピングを軽減するための軽量なアプローチとして、ボーン放射距離に基づく内外順序推定と、Kahn のアルゴリズムを用いた Vulkan トポロジカルディスパッチによるクリアランス補正を実装・評価しました。

### 検証の結論
**特定モデル・単一ポーズのCPU再現テストにおいて、定義した法線方向突出の閾値超過頂点数が75から0へ減少した。実GPU経路の動作・性能、三角形交差、他モデルや連続動作への一般化は未検証である。**

### 残された課題
- **三角形メッシュ交差の非保証**: 点-接平面拘束のみでは、エッジ同士の交差やシワの折り返しによる交差を完全に防ぐことはできません。
- **未バインド頂点の扱い**: 1.1% の未バインド頂点に対するフォールバックや、複数親サーフェス接触への対応。
- **実GPU環境での計測**: 実機描画パイプラインにおけるフレームタイムや同期オーバーヘッドの厳密なプロファイリング。
- **陰関数・SDFアプローチとの統合**: より高品位な接触解消が求められるケースにおいては、Buffetら (2019) のような連続距離場手法との併用が今後の検討課題となります。

---

*Copyright © 2026 VulVATAR Project. All rights reserved.*
