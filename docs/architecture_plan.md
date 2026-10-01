# 课题升级计划：评论路由层（LSTM 基线 + Qwen-0.6B 微调）双轨对比

课题：《商品评价的观点自动提取》
升级目标：把"线性脚本"升级为"分层 Agent 架构"，新增一个**评论路由层**，替代"每条评论都送大模型深挖"的低效做法。
核心手法：**同一任务，跑两条技术路线，做对比实验**，用数据证明"任务特性决定选型"。

---

## 一、要解决的问题

痛点：每条评论都调大模型，贵、慢；很多评论根本不需要现状（已有代码）：爬虫 → 每条评论都丢给 DeepSeek 双角色精析 → 打分 → 可视化。
深度分析。

升级后：评论先进**本地路由层**初筛，判断"这条值不值得送 DeepSeek 深挖"，只有该深挖的（水军/反讽/AI生成/极端评价）才调用大模型。

这套分工对应当下 Agent 架构的 "System One（快速直觉决策） + System Two（深度推理）"，并且**路由层是我们自研的**，不是调现成 API，含金量高。

---

## 二、双轨路由层：对比实验设计（核心）

同一份数据集、同一个任务（判断"评论是否可疑，需要深挖"），跑两条路线：

| | 路线 A：LSTM 基线 | 路线 B：Qwen3-0.6B 微调 |
|---|---|---|
| 定位 | System One 快速决策 | System Two 深度判断 |
| 参数量 | 极小（几十万级） | 0.6B（QLoRA） |
| 训练速度 | 秒级到分钟级 | 分钟级 |
| 推理速度 | 极快 | 较快 |
| 成本 | 接近零 | 低 |
| 精度 | 基线水平 | 更高 |
| 依赖 | torch + jieba | torch + transformers + peft + bitsandbytes |

**产出对比表**：两条路线的 准确率 / 单条耗时 / 参数量 / 大模型调用节省率。这是报告里的硬实验结果。

**叙事落点（梁文锋式克制）**：分析任务特性后发现是轻量短文本分类，无需一律堆大模型；通过对比证明在保证精度的前提下，轻量 RNN 能覆盖大部分场景，大模型微调只在极端样本上体现优势。用数据说话，不炫技。

---

## 三、整体架构

```
数据采集(爬虫 / MCP 工具层)
   ↓
【Router 层】(本课题新增，二选一或并联对比)
   ├─ LSTM 基线 ──────────┐
   └─ Qwen-0.6B 微调 ─────┤ → 判断"是否可疑，需要深挖"
   ↓ 分流
【DeepSeek 双角色精析】(只处理可疑/该深挖的，保留原有亮点)
   ↓
可视化 + 报告
```

---

## 四、环境现状（已探明）

- 目标机器：Windows 台式机 + WSL2（Ubuntu，Linux 6.6 WSL2），通过 SSH 远程操作
- GPU：GTX 1050 Ti，4GB 显存，驱动 560.94，CUDA 12.6（WSL 内）/ 系统装 CUDA 12.8 toolkit
- Python 3.12，磁盘 927G 空闲，内存 15G
- 已装：CUDA 12.8 toolkit、nvidia-smi
- 缺：torch、jieba、gensim（词向量）、训练框架
- **关键约束：1050 Ti 是 Pascal 架构（sm_61），torch 须装 CUDA 12.x 兼容版（约 2.4~2.6），不能装最新**

---

## 五、整体步骤

### 第 1 步：WSL 搭环境
- 建 venv
- 装 torch（Pascal 兼容版）+ transformers / accelerate / peft / bitsandbytes / datasets / jieba / gensim
- 验证 `torch.cuda.is_available() == True`，跑 GPU 冒烟测试

### 第 2 步：造数据集（两份模型共用，只做一次）
- 从现有豆瓣评论 CSV 取样，用 DeepSeek 弱标签 / 特征规则半自动标注
- 标签：`可疑(需深挖)` vs `普通(不需深挖)`
- 几百条即可，切 train/valid

### 第 3 步：路线 A，LSTM 基线
- **注意：RNN 路线不需要 tokenizer**。tokenizer 是 Transformer（路线 B Qwen）的概念；LSTM 用 **jieba 分词 + 词索引** 即可
- jieba 分词 → 查预训练词向量表的词表（vocab）得到词索引 → 喂 LSTM
- **预训练中文词向量**（gensim 加载，如腾讯/搜狗 word2vec 或 fastText 中文向量）初始化 embedding 层，词表由词向量文件自带
- **为什么用预训练而非随机初始化**：数据量少，预训练词向量自带词义，训练数据少也能 work 得好，且报告可写"采用预训练中文词向量初始化嵌入层"，更正规
- 网络：预训练 embedding → 双向 LSTM → 二分类
- 训练，记录 loss / 准确率 / 单条推理耗时

### 第 4 步：路线 B，Qwen3-0.6B 微调
- QLoRA（4bit + Lora）微调 0.6B
- 训练，记录 loss / 准确率 / 单条推理耗时

### 第 5 步：对比实验 + 接入框架
- 同一批测试集上跑两条路线，出对比表
- 把 Router 接进现有 mcp/ agent 框架（server.py + chat.py 那套增量）
- 保留 DeepSeek 双角色精析为深度层
- 统计大模型调用节省率，留真实运行日志

### 第 6 步：出结果、写报告
- 架构图、对比表、loss 曲线、准确率、成本/速度数据
- 更新原 570 行报告，突出分层架构 + 双轨对比

---

## 六、时间与风险

- 国庆假期时间充裕，分几天推进，不赶
- 风险：
  1. torch 版本不兼容 Pascal → 用查好的兼容版规避
  2. 数据集标注质量 → DeepSeek 弱标签 + 人工抽查
  3. 4G 显存 → 两条路线都够（LSTM 极小，Qwen 用 QLoRA）
  4. WSL 隧道断连 → 注意保持会话

---

## 七、版本管理与上云（GitHub 私有仓库）

**目标**：全程用 git 做版本管理，源码推到 GitHub 私有仓库。训练和开发在大电脑（WSL2）上进行，但主人不需要守在大电脑旁，用自己房间的 desktop 远程浏览 GitHub 仓库即可跟进进度、审代码。

### 7.1 仓库策略
- 仓库：**private**（必须私有，见 7.3 泄漏风险）
- 命名建议：`douban-review-router`（或按课题相关命名），放 GitHub 账号 Visol-456 下
- 工作流：
  1. 大电脑 WSL2 是**开发/训练机**，所有改动在本地 git 仓库提交
  2. 每次有可提交进展（环境搭好、数据集就绪、LSTM 训出、Qwen 训出、对比结果）就 `push` 到 GitHub
  3. 主人 desktop 只看 GitHub 仓库，`git log` / 文件浏览 / diff 都看得见，无需碰大电脑
- 源码、训练脚本、数据集（脱敏后）、结果图、报告草稿全部保留在仓库，随时可回退

### 7.2 防 API Key 泄漏（硬规矩，写进 .gitignore）
这份代码里出现过真实密钥（DeepSeek / OpenAI 兼容 / Winston AI），**绝不能进仓库**。规矩：
- 所有密钥统一放到**环境变量 / 单独的 `.env`**（`.env` 必须 gitignore）
- 代码里引用密钥一律 `os.environ.get("...")`，**不允许硬编码**
- 已有代码里硬编码的 key（如 pose.py / asr.py / main.py 里的），上仓库前全部抽出来替换成环境变量
- `.gitignore` 至少包含：
  ```
  .env
  *.key
  *secret*
  config.py   # 若含密钥
  *.csv       # 若 CSV 含用户信息（必要时脱敏后再入库）
  __pycache__/
  *.pt / *.safetensors / *.bin   # 模型权重不推仓库（太大），用 release / 单独说明
  ```
- **提交前自查**：`git grep -iE "sk-|api[_-]?key|secret|token"` 扫一遍，确保没有真实密钥残留
- CSV 若含真实评论数据，先脱敏（去掉用户名等个人信息）再入库，或只放脱敏样例

### 7.3 为什么必须 private
- 代码里历史版本可能残留密钥痕迹，private 仓库外人不可见，避免被薅
- 课题成果未定稿前不宜公开
- 后续如需公开（比如作为开源项目展示），再单独开 public 仓库并彻底清洗密钥

### 7.4 主人远程跟进方式
- 在房间 desktop 上：浏览器开 GitHub 仓库 → 看 commit 历史、文件、最新进度
- 需要时我可以把关键文件（训练曲线图、对比表、报告草稿）也 push 进仓库或单独发
- 大电脑只负责跑训练，主人不用守在旁边

---

## 八、待确认

1. 双轨对比这个结构，你认可吗？（我强烈推荐，报告最有含金量的写法）
2. 数据集用现有豆瓣评论 + DeepSeek 弱标签，行吗？
3. 报告保留原"双角色评估"作深度层，行吗？
4. LSTM 用 jieba 分词 + gensim 词向量（经典做法，报告正规），认可吗？

---

## 九、Agent 架构升级：Tools / MCP / Skills 三能力（本次新增）

在原「Router → 只分流可疑样本 → 大模型深挖」的基础上，主 agent
（`src/agent/agent_chat.py`）新增三大能力，让深度分析从「模型裸猜」升级为
「**先查真实口碑，再结合文本判断**」。三者的分工：

```
                    ┌───────────────────────────────┐
   用户/评论  ────► │  主 Agent (agent_chat.py)        │
                    │  system prompt = 基础 + Skill    │
                    └───────────────┬───────────────┘
                                    │ 自主决定调用哪个工具
              ┌─────────────────────┼──────────────────────┐
              ▼                     ▼                      ▼
        【MCP 工具】           【Tools 搜索】           【Skills 知识】
   search_movie_info      deepseek_web_search       movie_review_analyst.md
   run_python_code        metaso_search             （注入 system prompt）
   （FastMCP stdio）        （function tools）
              │                     │
              └──────► 真实口碑基准 ◄┘
                          │
                          ▼
              模型判断：可疑 0/1 + 依据
```

### 9.1 MCP：可执行工具（含查电影口碑）

- **落点**：`src/agent/mcp_server.py`，用 FastMCP 起一个 stdio 服务，`agent_chat.py` 通过
  `mcp.client.stdio.stdio_client` 拉起并 `session.list_tools()` 拿到工具清单。
- **工具 1 `search_movie_info(movie_title)`**：调用豆瓣移动端 rexxar 搜索接口
  `GET https://m.douban.com/rexxar/api/v2/search?q=<title>&type=movie`，
  解析 `subjects.items[].target`，返回**片名 / 年份 / 类型主创 / 豆瓣评分 / 评分人数 / 豆瓣ID**。
  这是给模型判断评论可信度的**外部真实口碑基准**（例如「狂飙」8.5 分、107 万人评）。
  匿名接口，不涉及任何密钥。
- **工具 2 `run_python_code(code)`**：模型可写 Python 做数据清洗/统计。
- **工具转换**：`_convert_tool()` 把 MCP 工具对象转成 OpenAI function-calling 的
  `{"type":"function","function":{name,description,parameters}}` 结构，与 Tools 合并成一份 tools 列表。

### 9.2 Tools：模型自主调用的联网搜索

保留原有 **OpenAI 兼容 `chat.completions` + `tools` 循环**（`tool_choice="auto"`、流式解析
`delta.tool_calls`），并新增两个 function tools 供模型自主调用：

- **`deepseek_web_search(query)`**：接入 **DeepSeek Responses API 自带的服务端联网搜索**。
  请求 `POST {BASE_URL}/v1/responses`，`tools=[{"type":"web_search"}]`，
  并加护栏 `max_tool_calls=3` + `reasoning.effort=low`，避免多轮搜索烧上下文。
  返回模型搜索后的带链接结论。
- **`metaso_search(query)`**：秘塔 Metaso 第三方搜索兜底
  （`POST https://metaso.cn/api/v1/search`，`Authorization: Bearer $METASO_API_KEY`），
  返回 `webpages[]` 的标题/链接/摘要。

**执行分发**：工具调用时，`deepseek_web_search` / `metaso_search` 走本地
`_EXTRA_TOOL_HANDLERS`，其余（MCP 工具）走 `session.call_tool()`。结果以 `role:"tool"`
消息回灌，模型继续推理，直到不再发起工具调用。

### 9.3 Skills：注入领域专家知识

- **落点**：`skills/movie_review_analyst.md`（「影评分析专家」skill）。
- **内容**：反讽/阴阳怪气识别、水军刷分识别、**结合真实口碑对照判断**、
  极端情绪/引战/AI 生成识别，以及标准工作流（提取可核验主张 → 调用外部查询 →
  逐维度打分 → 输出 `suspicion` / `needs_deep_dive` / `reason`）。
- **注入方式**：`_build_system_prompt()` 在每次会话启动时读取该文件，拼进
  `system_prompt` 的「注入的领域专家 Skill」区块，`messages[0]` 即为 system prompt。
  报告可写：**「系统通过 Skills 注入领域专家知识」**。

### 9.4 端到端流程（一条评论进来）

1. Router（LSTM / Qwen）判定该评论是否可疑 → 命中才进深挖。
2. 主 Agent 收到评论 + Skill 注入的判定准则。
3. 模型**先调用 `search_movie_info`**（或 `deepseek_web_search` / `metaso_search`）查该片真实口碑。
4. 把「评论文本」与「真实口碑」对照，按 Skill 准则给出可疑度与依据。
5. 输出结构化结论（可疑 0/1 + reason）。

### 9.5 防泄漏

- 新增代码全部 `os.environ.get(...)` 读密钥：`DEEPSEEK_API_KEY` / `METASO_API_KEY` /
  `DEEPSEEK_SEARCH_MODEL`，**无任何硬编码**。
- 豆瓣搜索为匿名公开接口，不带密钥。
- 提交前 `git grep` 扫描确认无泄漏。
