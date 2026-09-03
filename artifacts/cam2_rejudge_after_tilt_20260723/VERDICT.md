# cam2 after raise+tilt — 2026-07-23 17:22

## Verdict
**未满足 agentview 要求（方向对，幅度/机位仍偏）。**

抬高+下倾后，画面更俯视、夹爪进入上沿，但相对 LIBERO agentview 的定量指标 **比调整前更远**。

## Metrics (scene 224 crop)

| | brightness | mass_cy | lower−upper |
|--|--|--|--|
| train | 71.1 | **81.9** | **−34.0** |
| before | 111.4 | 131.8 | +39.6 |
| **after tilt** | 109.4 | **141.9** ↑ | **+78.1** ↑ |

Closer to train would move `mass_cy` ↓ toward ~82 and `lower−upper` toward negative.

## Qualitative
- After: 近乎俯视桌面，笔居中，夹爪在画面上方 — 有“下倾”感。
- Train agentview: 更高更远的斜俯视，上半有空间/臂身，下半工作区，不是贴桌俯拍。
- Domain gap (木桌 vs sim) 仍在，预期内。

## Recommendation
再调 cam2：**抬高并略后移**，下倾保持或略减，使桌面不要占满下半、上半露出更多臂身/空间；目标让 `mass_cy` 降到 ~90–110、`lower−upper` 降到接近 0 或负值。然后再判一次。
