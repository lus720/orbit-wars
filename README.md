# Orbit Wars Agent

conda activate kaggle

## 服务器








### 3. 提交到 Kaggle

```bash
kaggle competitions submit -c orbit-wars -f main.py -m "submit smart_agent"
```

### 4. 查询提交状态

```bash
/home/ls/miniconda3/envs/kaggle/bin/kaggle competitions submissions -c orbit-wars -v
```

### 5. 查看比赛页面 API 内容

```bash
curl -L 'https://www.kaggle.com/api/v1/competitions/orbit-wars/pages'
```

## 安装依赖

安装本项目训练、评估、回放和 Notebook 常用的 Python 库：

```bash
python -m pip install -U pip numpy pyyaml torch kaggle-environments kaggle jupyter ipykernel
```

如果你使用的是当前 conda 环境，也可以显式指定 Python：

```bash
/home/ls/miniconda3/envs/kaggle/bin/python -m pip install -U pip numpy pyyaml torch kaggle-environments kaggle jupyter ipykernel
```

## 相关命令

这一节尽量记录本项目实际会用到、并且我已经在当前工作流里用过的命令，保证用户可以手动复现。

### 本地训练 PPO

```bash
/home/ls/miniconda3/envs/kaggle/bin/python -m src.train --config default_cfg.yaml
```

训练 checkpoint 会保存到 `artifacts/orbit_wars_ppo/`。如果你看到 `PermissionError: [Errno 13] Permission denied: '/kaggle'`，说明配置里的 `save_dir` 指向了 Kaggle Notebook 专用目录；本地运行时应使用类似 `artifacts` 这样的相对路径。

### 运行前的网络说明

如果你的机器和当前环境一样依赖代理访问 Kaggle，可以先检查代理变量：

```bash
env | rg '^(http_proxy|https_proxy|HTTP_PROXY|HTTPS_PROXY|KAGGLE)'
```

如果需要临时取消代理直连，可以用：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy curl -I https://api.kaggle.com
```

如果需要测试当前代理是否能访问 Kaggle，可以用：

```bash
curl -I https://api.kaggle.com
```

### 查看本地文件

```bash
ls -1 /home/ls/kaggle
```

```bash
sed -n '1,220p' /home/ls/kaggle/main.py
```

```bash
sed -n '1,260p' /home/ls/kaggle/README.md
```

```bash
sed -n '1,260p' /home/ls/kaggle/STRATEGY.md
```

### 查看 conda 环境

```bash
conda env list
```

### 查看 kaggle 环境里的 Jupyter 组件

```bash
/home/ls/miniconda3/envs/kaggle/bin/python -m jupyter --version
```

### 查看比赛文件

```bash
/home/ls/miniconda3/envs/kaggle/bin/kaggle competitions files -c orbit-wars
```

### 查看排行榜

```bash
/home/ls/miniconda3/envs/kaggle/bin/kaggle competitions leaderboard -c orbit-wars
```

### 查看环境配置

```bash
/home/ls/miniconda3/envs/kaggle/bin/python -c "from kaggle_environments import make; env=make('orbit_wars', debug=True); print(env.configuration)"
```

### 查看 Kaggle CLI 版本

```bash
/home/ls/miniconda3/envs/kaggle/bin/kaggle --version
```

### 查询比赛提交历史

```bash
/home/ls/miniconda3/envs/kaggle/bin/kaggle competitions submissions -c orbit-wars -v
```

### 提交本地 agent

```bash
/home/ls/miniconda3/envs/kaggle/bin/kaggle competitions submit -c orbit-wars -f /home/ls/kaggle/main.py -m "submit smart_agent"
```

### 拉取比赛 overview 的原始 HTML

这个命令拿到的是前端壳页面，不一定包含正文，但手动排查网页结构时有用：

```bash
curl -L https://www.kaggle.com/competitions/orbit-wars/overview
```

### 拉取比赛 pages API

这是目前最有用的接口之一，可以直接拿到比赛各个页面的内容，包括 `How to Play Orbit Wars`：

```bash
curl -L 'https://www.kaggle.com/api/v1/competitions/orbit-wars/pages'
```

### 只提取 `How to Play Orbit Wars` 页面内容

如果你想手动把规则页单独抽出来，可以用 Python 过滤：

```bash
curl -L 'https://www.kaggle.com/api/v1/competitions/orbit-wars/pages' | /home/ls/miniconda3/envs/kaggle/bin/python - <<'PY'
import json, sys
data = json.load(sys.stdin)
for page in data["pages"]:
    if page["name"] == "How to Play Orbit Wars":
        print(page["content"])
        break
PY
```

### 查看 pages API 中有哪些页面

```bash
curl -L 'https://www.kaggle.com/api/v1/competitions/orbit-wars/pages' | /home/ls/miniconda3/envs/kaggle/bin/python - <<'PY'
import json, sys
data = json.load(sys.stdin)
for page in data["pages"]:
    print(page["name"])
PY
```

### 直接运行 notebook 中的代码单元

```bash
/home/ls/miniconda3/envs/kaggle/bin/python - <<'PY'
import json
from pathlib import Path

nb = json.loads(Path('orbit-wars-solution.ipynb').read_text())
ctx = {'__name__': '__main__'}
for idx, cell in enumerate(nb['cells']):
    if cell.get('cell_type') != 'code':
        continue
    code = ''.join(cell.get('source', []))
    print(f'>>> executing cell {idx}')
    exec(compile(code, f'orbit-wars-solution.ipynb:cell_{idx}', 'exec'), ctx)
print('NOTEBOOK_EXECUTION_OK')
PY
```

### 快速测试 `smart_agent`

```bash
/home/ls/miniconda3/envs/kaggle/bin/python - <<'PY'
from kaggle_environments import make
from main import smart_agent

env = make("orbit_wars", debug=True)
env.run([smart_agent, "starter"])
for i, s in enumerate(env.steps[-1]):
    print(i, s.reward, s.status)
PY
```

### 检查 `main.py` 语法

```bash
/home/ls/miniconda3/envs/kaggle/bin/python -m py_compile /home/ls/kaggle/main.py
```

### 跑一组基准对战回归

```bash
/home/ls/miniconda3/envs/kaggle/bin/python - <<'PY'
from statistics import mean
from kaggle_environments import make
from main import smart_agent

def evaluate(agents, games=4):
    rewards = []
    for _ in range(games):
        env = make('orbit_wars', debug=True)
        env.run(agents)
        rewards.append(env.steps[-1][0].reward)
    return rewards

for name, agents in [
    ('2p vs random', [smart_agent, 'random']),
    ('2p vs starter', [smart_agent, 'starter']),
    ('4p vs random', [smart_agent, 'random', 'random', 'random']),
    ('4p vs starter', [smart_agent, 'starter', 'starter', 'starter']),
]:
    rewards = evaluate(agents, games=4)
    print(name, rewards, mean(rewards))
PY
```

### 查看当前目录下 notebook 的前几个单元

```bash
python - <<'PY'
import json
from pathlib import Path
nb = json.loads(Path('orbit-wars-solution.ipynb').read_text())
for i, cell in enumerate(nb['cells'][:6]):
    print(i, cell['cell_type'])
    print(''.join(cell.get('source', []))[:400])
    print()
PY
```

## 进一步阅读

更完整的说明见 [STRATEGY.md](/home/ls/kaggle/STRATEGY.md)：

- 完整规则拆解
- 当前策略逐段解释
- 本地测试命令
- Kaggle CLI 提交命令
- 当前策略局限和后续优化方向
