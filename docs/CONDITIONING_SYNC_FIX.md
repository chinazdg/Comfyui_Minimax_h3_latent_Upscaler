# Conditioning 同步修复说明 / Conditioning Sync Fix

- **改动文件 / Modified File**：`nodes/minimax_h3_latent_upscaler_3d.py`
- **改动分支 / Branch**：`feature/conditioning-sync`
- **上游仓库 / Upstream**：https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
- **本 Fork / This Fork**：https://github.com/chinazdg/Comfyui_Minimax_h3_latent_Upscaler
- **日期 / Date**：2026-08-28

---

## 1. 修复了什么问题 / What This Fixes

### 1.1 报错现象 / Symptom

在使用 MiniMax H3 做**多片段衔接生成**时（Motion Context 续写、或首尾帧 keyframe 续写），
只要中间插入了本插件的 3D 潜空间放大节点，就会出现：

- 第 1 段：正常生成 + 正常放大 ✅
- 第 2 段：一进采样就崩溃 ❌

```
RuntimeError: shape mismatch:
value tensor of shape [5505, 96] cannot be broadcast to indexing result of shape [6043, 96]

  File "comfy\ldm\minimax\model.py", line 654, in _forward
    all_video_rows[~img_update] = cond_video_rows
```

同样的报错在使用首尾帧（I2VA/FL2VA）和 Motion Context 两种模式下都会触发，
说明问题**不在具体续写方式**，而在放大节点本身。

### 1.2 根本原因 / Root Cause

崩溃点在 H3 模型内部 `comfy/ldm/minimax/model.py` 第 644–655 行：

```python
img_update     = layout.img_update.to(device)          # 掩码，长度 = 当前 latent 的行数
video_rows     = patchify_video(video_x, ...)          # 当前待生成的 latent 行
cond_video_rows = self._cond_video_rows(payload, ...)  # 从 conditioning 取出的历史帧行

all_video_rows = torch.empty(img_update.shape[0], video_rows.shape[1], ...)
all_video_rows[~img_update] = cond_video_rows   # ← 第 654 行，崩溃点
all_video_rows[img_update]  = video_rows
```

这行代码的语义是：**把 conditioning 里缓存的历史帧（上一片段的尾帧 / Motion Context）
填进"本次不重新生成"的帧位**。它要求两侧行数严格相等。

而 latent 的行数由分辨率决定（H3 VAE 空间压缩 16×，再按 patch 展开）：

| 阶段 | 分辨率 | latent | 行数 |
|------|--------|--------|------|
| 第 1 段生成 | 512×896 | 32×56 | ≈ 5505 |
| 3D 放大之后 | 768×1344 | 48×84 | ≈ 6043 |

问题就在这里：

- 3D 放大节点**只放大了 latent**（5505 → 6043 行）
- 但 **conditioning 里的元数据没动**，仍然记录 `latent_h=32, latent_w=56`，
  并且还挂着按旧尺寸算出来的 `minimax_keyframes`（首尾帧）和 `minimax_refs`（参考帧）

于是第 2 段进来时：

- `img_update.shape[0]` = 6043（按**放大后**的 latent 算）
- `cond_video_rows` = 5505（从**未同步**的 conditioning 里取）
- 6043 ≠ 5505 → `shape mismatch` 崩溃

**一句话总结**：放大节点改变了 latent 的空间尺寸，却没有同步更新 conditioning 中
记录的尺寸元信息和历史帧张量，导致两者在数据管线里错位。

> 社区已知约束：H3 官方节点文档明确写过 "Latent motion context cannot resize."
> 即 motion context 的 latent 不能跨分辨率重采样。这正是冲突的源头。

---

## 2. 改动内容 / What Changed

### 2.1 改动概览

| 位置 | 改动类型 | 说明 |
|------|---------|------|
| `_process_conditioning()` | **新增函数** | conditioning 同步核心逻辑（约 78 行） |
| `define_schema().inputs` | **新增 2 个入参** | `conditioning`（可选）、`conditioning_mode`（三选一） |
| `define_schema().outputs` | **新增 1 个出参** | `conditioning`（同步后的结果） |
| `execute()` 签名 | **扩展** | 增加 `conditioning=None, conditioning_mode="NO_refs"` |
| `execute()` 返回 | **改为 tuple** | 从 `io.NodeOutput(...)` 改为 `(latent_dict, conditioning)` |
| 早退分支 | **同步修改** | 尺寸未变时也返回同步后的 conditioning |

### 2.2 新增的 `conditioning_mode` 三档

| 模式 | 行为 | 适用场景 | 副作用 |
|------|------|---------|--------|
| `pass_through` | 完全不碰 conditioning，保持原节点行为 | 单段放大、不需要衔接 | 多段衔接仍会崩（用于对照验证） |
| `NO_refs` | 更新 `latent_h/latent_w`，**删除** `minimax_keyframes` 与 `minimax_refs` | 各段内容独立，只需风格一致 | 丢失跨段视觉连续性 |
| `refs` | 更新 `latent_h/latent_w`，并把历史帧**插值放大**到新尺寸 | 需要严格运动/画面连续 | 插值可能引入轻微重影（ghosting） |

### 2.3 核心代码

```python
def _process_conditioning(conditioning, latent_height, latent_width, mode):
    """Sync conditioning metadata with upscaled latent dimensions.

    Ported from supElement/ComfyUI_Element_easy/MinimaxH3LatentUpscaler_Adv.py
    """
    if conditioning is None or mode == "pass_through":
        return conditioning

    out = []
    for entry in conditioning:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            out.append(entry)
            continue

        emb, meta = entry[0], entry[1]
        new_meta = meta.copy()

        # 关键：把 conditioning 记录的尺寸改写成放大后的尺寸
        new_meta["latent_h"] = latent_height
        new_meta["latent_w"] = latent_width

        if mode == "NO_refs":
            # 删掉按旧尺寸算出来的历史帧，让模型跳过混合逻辑
            new_meta.pop("minimax_refs", None)
            new_meta.pop("minimax_keyframes", None)

        elif mode == "refs":
            # 把历史帧插值到新尺寸，保留连续性
            def upscale_lat_dict(d):
                if not isinstance(d, dict):
                    return d
                new_d = d.copy()
                t = d.get("latent")
                if isinstance(t, torch.Tensor):
                    if len(t.shape) == 4:      # 图片参考 (B, C, H, W)
                        new_d["latent"] = F.interpolate(
                            t, size=(latent_height, latent_width),
                            mode='bicubic', align_corners=False)
                    elif len(t.shape) == 5:    # 视频参考 (B, C, T, H, W)
                        b, c, tf, h, w = t.shape
                        t_flat = t.permute(0, 2, 1, 3, 4).contiguous().view(-1, c, h, w)
                        ups = F.interpolate(
                            t_flat, size=(latent_height, latent_width),
                            mode='bicubic', align_corners=False)
                        nh, nw = ups.shape[-2], ups.shape[-1]
                        new_d["latent"] = ups.view(b, tf, c, nh, nw).permute(0, 2, 1, 3, 4)
                new_d["latent_h"] = latent_height
                new_d["latent_w"] = latent_width
                return new_d

            for key in ["minimax_refs", "minimax_keyframes"]:
                val = meta.get(key)
                if val is not None and isinstance(val, list):
                    new_meta[key] = [upscale_lat_dict(item) for item in val]

        out.append([emb, new_meta])

    return out
```

### 2.4 为什么 `NO_refs` 能修好崩溃

H3 的 `model.py` 第 652 行有一个判空：

```python
if cond_video_rows is not None:
    all_video_rows = torch.empty(img_update.shape[0], ...)
    all_video_rows[~img_update] = cond_video_rows   # 崩溃点
    all_video_rows[img_update]  = video_rows
```

`NO_refs` 把 `minimax_keyframes` 和 `minimax_refs` 删掉之后，
`_cond_video_rows()` 返回 `None`，**整个混合分支被跳过**，
直接用放大后的 latent 走正常生成流程 —— 维度冲突自然消失。

代价是第 2 段不再继承第 1 段的画面记忆，所以只适合"各段内容独立"的场景
（例如不同模特各自的走秀镜头）；如果需要严格连续，改用 `refs` 模式。

### 2.5 顺带修掉的第二个 bug：返回值格式

第一版改造把返回值写成了：

```python
return io.NodeOutput({"samples": out}), synced_conditioning   # ❌
```

结果下游节点报：

```
ValueError: dictionary update sequence element #0 has length 1; 2 is required
  File "comfy_extras\nodes_lt.py", line 796, in execute
    output.update(video_latent)
```

原因：`io.NodeOutput` 是新 API 用于**声明/类型提示**的包装类，
实际执行时多输出节点应该返回**裸 tuple**。已修正为：

```python
return ({"samples": out}, synced_conditioning)   # ✅
return (latent, synced_conditioning)             # ✅ 早退分支
```

---

## 3. 对原有功能的影响 / Compatibility

- **向后兼容**：`conditioning` 是可选输入，不接时行为与原节点完全一致；
  `conditioning_mode` 默认 `NO_refs`，但 conditioning 为 `None` 时自动退化成 `pass_through`。
- **放大质量不变**：没有改动神经网络结构、归一化参数、对齐算法、时间分块逻辑，
  学习式上采样的画质与显存占用保持原样。
- **新增开销**：`NO_refs` 模式几乎零开销（只改字典）；
  `refs` 模式会对历史帧做一次 bicubic 插值，开销很小。

---

## 4. 工作流接线 / Wiring

放大节点必须放在**采样之前**，同步后的 conditioning 才能被采样器用上：

```
首帧图 → MiniMaxH3ImageToVideo ─┬─ conditioning (原始分辨率) ─┐
                                └─ latent (原始分辨率) ────────┤
                                                              ↓
                                        Minimax H3 Latent Upscaler (3D)【改造版】
                                              conditioning_mode = NO_refs
                                                              ↓
                                        ┌─────────────────────┴────────────────────┐
                                        ↓                                          ↓
                                latent (已放大)                          conditioning (已同步)
                                        ↓                                          ↓
                                        └──────────→ BasicGuider ←─────────────────┘
                                                         ↓
                                        SamplerCustomAdvanced ← RandomNoise
                                                         ↓
                                                    VAE Decode
```

关键点：

1. 同步后的 `conditioning` 接 `BasicGuider.conditioning`
2. 放大后的 `latent` 接 `SamplerCustomAdvanced.latent_image`
3. 不要再接一个未同步的 `MiniMaxH3ImageToVideo`，否则又把旧尺寸的 conditioning 灌回去

---

## 5. 验证方式 / Verification

运行后控制台应出现：

```
[MinimaxH3-3D] Latent 32x56 -> 48x84 | Pixels 768x1344 | scale=1.500
[MinimaxH3-3D] Conditioning synced: 84x48 latent | mode=NO_refs (removed keyframes/refs)
[MinimaxH3-3D] ✅ Model offloaded to CPU. VRAM released for next node.
```

- ✅ 出现以上三行且第 2 段正常出片 → 修复生效
- ❌ 仍报 `shape mismatch` → 检查 conditioning 是否真的接到了放大节点
- ❌ 报 `dictionary update sequence` → 节点未重新加载，清 `__pycache__` 后重启

---

## 6. 已知限制 / Known Limitations

1. `NO_refs` 会丢弃跨段视觉连续性，需要连续镜头时请改用 `refs` 并人工检查重影。
2. `refs` 使用通用 bicubic 插值放大历史帧，未针对 H3 潜空间分布做学习式放大，
   长链条衔接可能出现轻微 ghosting。
3. 更彻底的方案是让 H3 内核支持跨分辨率 motion context 重采样，
   那需要改动 `comfy/ldm/minimax/model.py`，超出本节点范围。
