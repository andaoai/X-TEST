---
name: env-manager
description: Python uv 环境管理。添加/删除依赖、查看包、锁定版本、Python 版本管理。
user-invocable: true
allowed-tools: Read Write Edit Bash Grep Glob
---

# Python 环境管理

基于 uv 管理项目的 Python 依赖。

## 项目信息

- **Python**: 3.13.3 (`.python-version`)
- **uv**: 0.7.21
- **虚拟环境**: `.venv/`
- **依赖文件**: `pyproject.toml` + `uv.lock`

## 命令

### 查看依赖

```bash
uv pip list                          # 已安装的包
uv tree                              # 依赖树
cat pyproject.toml                   # 查看声明依赖
```

### 添加依赖

```bash
uv add <package>                     # 添加到 pyproject.toml 并安装
uv add <package>=<version>           # 指定版本
uv add <package> --dev               # 开发依赖
```

### 删除依赖

```bash
uv remove <package>                  # 从 pyproject.toml 移除并卸载
```

### 更新依赖

```bash
uv lock --upgrade <package>          # 升级单个包
uv sync                              # 同步所有依赖到最新兼容版本
```

### 运行脚本

```bash
uv run python <script.py>            # 在 .venv 中运行
uv run python run.py --algo gabor_lift
```

### 清理

```bash
uv cache clean                       # 清理 uv 缓存
```

## 当前依赖

| 包 | 用途 |
|---|------|
| matplotlib | 可视化 |
| numpy | 数值计算 |
| pillow | 图像渲染 |
| scikit-learn | t-SNE、PCA |
| scipy | Gabor 卷积 |
| torch / torchvision | 神经网络 (可选) |
| roboflow | 下载 Roboflow 数据集 |
| tqdm | 进度条 |

## 注意事项

- **必须使用 `uv run` 运行脚本**，不能直接 `python xxx.py`
- 加包时用 `uv add` 而不是 `pip install`，否则不会更新 `pyproject.toml`
- 如果 `uv lock` 卡住，尝试 `UV_HTTP_TIMEOUT=300 uv add <package>`
