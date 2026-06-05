---
name: dataset-manager
description: 数据集管理。查看、添加、预览、下载数据集。管理 data/ 目录和 experiments/source/ 中的所有数据源。
user-invocable: true
allowed-tools: Read Write Edit Bash Grep Glob
---

# Dataset Manager

管理项目中的所有数据集。

## 相关目录

```
data/                         ← 原始数据文件 (CIFAR-10, COCO 等)
experiments/source/           ← 数据源代码
├── manage.py                 #   数据集管理入口
├── synthetic/                #   合成数据
│   ├── config.py             #     合成数据定义 (字母/颜色/位置)
│   └── data.py               #     Dataset 生成器
└── real/                     #   真实数据集
    ├── download.py           #     下载公开数据集 (URL)
    ├── download_roboflow.py  #     下载 Roboflow 数据集
    └── explore.py           #     浏览数据集结构和标签
```

## 命令

### 查看所有数据集

```bash
uv run python experiments/source/manage.py list
```

列出所有已注册的数据集：名称、描述、路径、存在状态。

### 查看数据集详情

```bash
uv run python experiments/source/manage.py show <name>
```

显示数据集的完整信息和实际文件统计。

### 快速预览

```bash
uv run python experiments/source/manage.py preview synthetic_text    # 合成文字样本
uv run python experiments/source/manage.py preview cifar10          # CIFAR-10 样本
uv run python experiments/source/manage.py preview coco128-seg      # 实例分割样本
```

生成预览图保存到 `data/` 下。

### 扫描添加新数据集

```bash
uv run python experiments/source/manage.py add <path>
```

扫描目录，统计图片/标签/YAML 数量，给出注册表模板。用户在 `manage.py` 的 `DATA_REGISTRY` 中添加条目完成注册。

### 浏览真实数据集

```bash
uv run python experiments/source/real/explore.py
```

遍历 `data/instance_seg/` 下的所有数据集，打印结构、标签格式、生成样本预览。

### 下载

```bash
uv run python experiments/source/real/download.py           # 公开 URL 数据集
uv run python experiments/source/real/download_roboflow.py  # Roboflow 数据集
```

### 修改合成数据配置

编辑 `experiments/source/synthetic/config.py`：
- `EN_LETTERS` / `ZH_CHARS` — 文字列表
- `COLORS` — 颜色列表和 RGB 值
- `POSITIONS` — 位置名称和坐标

## 当前已注册数据集

| 名称 | 类型 | 描述 |
|------|------|------|
| synthetic_text | 合成 | 10英+10中 x 8色 x 5位 = 800张 |
| cifar10 | 真实 | CIFAR-10: 60000张 32x32, 10类 |
| coco128-seg | 真实 | COCO128: 128张, 80类, 实例分割 |
| crack-seg | 真实 | 裂缝分割: 4029张, 道路裂缝 |
| package-seg | 真实 | 包装分割: 2197张, 箱子/包裹 |
