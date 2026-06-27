# ML Intern 项目复现计划 - 8卡4090环境

> 基于项目深度分析，制定完整的全流程复现方案
> 硬件环境：8x NVIDIA RTX 4090 (24GB VRAM each, 192GB total)

---

## 目录

1. [项目本质与资源需求分析](#1-项目本质与资源需求分析)
2. [复现阶段规划](#2-复现阶段规划)
3. [详细实施步骤](#3-详细实施步骤)
4. [关键挑战与解决方案](#4-关键挑战与解决方案)
5. [学习路径建议](#5-学习路径建议)

---

## 1. 项目本质与资源需求分析

### 1.1 项目本质

**核心定位**：ML Intern 是一个 **Agent 框架**，不是训练框架。

**关键组件**：
```
┌─────────────────────────────────────────────────────────────┐
│                    ML Intern 架构                           │
├─────────────────────────────────────────────────────────────┤
│  1. Agent 框架 (核心)                                       │
│     - 队列驱动的异步事件系统                                 │
│     - Agentic Loop (最大 300 次迭代)                        │
│     - 上下文管理 (自动压缩)                                  │
│     - 工具系统 (本地 + 远程)                                 │
│                                                             │
│  2. LLM 调用 (通过 API)                                     │
│     - HF Router (router.huggingface.co)                    │
│     - OpenAI / Anthropic / 本地模型                         │
│     - 通过 LiteLLM 统一接口                                 │
│                                                             │
│  3. 工具生态                                                │
│     - 本地工具: bash, read, write, edit                     │
│     - 研究工具: papers, docs, github                        │
│     - HF 工具: jobs, repos, datasets                       │
│                                                             │
│  4. 后训练数据收集                                          │
│     - 会话轨迹记录                                         │
│     - SFT 数据导出                                         │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 资源需求分析

**关键发现**：项目本身 **不做训练**，只是调用 LLM API。

**资源需求分解**：

| 组件 | GPU 需求 | CPU/内存需求 | 存储需求 |
|------|---------|-------------|---------|
| **Agent 框架** | 无 | 2-4 CPU, 4-8GB RAM | 1GB |
| **本地 LLM 推理** | 1-2x 4090 (7B) | 4-8 CPU, 16-32GB RAM | 10-50GB |
| **SFT 训练** | 4-8x 4090 | 8-16 CPU, 64-128GB RAM | 100GB+ |
| **MongoDB** | 无 | 2-4 CPU, 4-8GB RAM | 10GB+ |
| **前端** | 无 | 1-2 CPU, 2-4GB RAM | 500MB |

### 1.3 8卡4090的利用策略

**策略**：将 8 卡 4090 分配给不同任务

```
8x 4090 分配方案：

方案 A：专注本地 LLM 推理
├── 4x 4090: 运行 70B 模型 (量化)
├── 2x 4090: 运行 13B 模型
└── 2x 4090: 备用/测试

方案 B：专注 SFT 训练
├── 8x 4090: 训练 7B-13B 模型
└── 使用 DeepSpeed ZeRO-3

方案 C：混合方案（推荐）
├── 2x 4090: 本地 LLM 推理 (7B-13B)
├── 4x 4090: SFT 训练
└── 2x 4090: 备用/测试
```

---

## 2. 复现阶段规划

### 阶段 1：基础框架搭建（无需 GPU，1-2天）

**目标**：搭建可运行的 Agent 框架

**任务**：
1. 环境配置
2. 依赖安装
3. 配置文件设置
4. 基础功能验证

**产出**：
- 可运行的 CLI Agent
- 使用 HF Router API 的基本对话

### 阶段 2：本地 LLM 集成（使用 1-2 卡，2-3天）

**目标**：集成本地 LLM，减少 API 依赖

**任务**：
1. 本地模型部署（vLLM/Ollama）
2. 集成到 Agent 框架
3. 性能测试和优化

**产出**：
- 本地 7B/13B 模型可对话
- 集成到 ml-intern 框架

### 阶段 3：SFT 训练流程（使用 4-8 卡，3-5天）

**目标**：实现完整的 SFT 训练流程

**任务**：
1. 数据准备
2. 训练脚本编写
3. 训练执行和监控
4. 模型评估

**产出**：
- 微调后的模型
- 训练日志和指标

### 阶段 4：完整流程验证（全流程，2-3天）

**目标**：验证完整的 Agent 工作流

**任务**：
1. 端到端测试
2. 性能优化
3. 文档整理

**产出**：
- 完整的复现文档
- 性能基准测试结果

---

## 3. 详细实施步骤

### 3.1 阶段 1：基础框架搭建

#### Step 1.1：环境准备

```bash
# 1. 克隆项目
cd /data/home/yizhou
git clone https://github.com/huggingface/ml-intern.git
cd ml-intern

# 2. 创建虚拟环境
uv sync

# 3. 安装 CLI 工具
uv tool install -e .

# 4. 验证安装
ml-intern --help
```

#### Step 1.2：配置文件设置

```bash
# 创建 .env 文件
cat > .env << EOF
HF_TOKEN=your_huggingface_token
GITHUB_TOKEN=your_github_token
EOF

# 验证配置
ml-intern "hello"
```

#### Step 1.3：基础功能测试

```bash
# 测试基本对话
ml-intern "What is 2+2?"

# 测试工具调用
ml-intern "List files in current directory"

# 测试研究功能
ml-intern "Research the latest developments in LLM agents"
```

**预期结果**：
- Agent 能正常响应
- 工具调用正常工作
- 研究功能能返回结果

---

### 3.2 阶段 2：本地 LLM 集成

#### Step 2.1：本地模型部署

**方案 A：使用 vLLM（推荐）**

```bash
# 1. 安装 vLLM
pip install vllm

# 2. 启动 vLLM 服务（7B 模型，单卡）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# 3. 启动 vLLM 服务（13B 模型，双卡）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-13B-Instruct \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --port 8001
```

**方案 B：使用 Ollama**

```bash
# 1. 安装 Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. 拉取模型
ollama pull qwen2.5:7b
ollama pull qwen2.5:13b

# 3. 启动服务
ollama serve
```

#### Step 2.2：集成到 Agent 框架

```bash
# 使用本地模型运行
ml-intern --model vllm/Qwen/Qwen2.5-7B-Instruct "your prompt"

# 或者使用 Ollama
ml-intern --model ollama/qwen2.5:7b "your prompt"
```

#### Step 2.3：性能测试

```python
# 测试脚本
import time
import requests

def test_local_llm(prompt, model="qwen2.5:7b"):
    start = time.time()
    response = requests.post("http://localhost:11434/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False
    })
    latency = time.time() - start
    return {
        "latency": latency,
        "tokens_per_second": len(response.json()["response"]) / latency
    }

# 测试不同规模的 prompt
prompts = [
    "Hello",  # 短 prompt
    "Write a Python function to sort a list",  # 中等 prompt
    "Explain the Transformer architecture in detail"  # 长 prompt
]

for prompt in prompts:
    result = test_local_llm(prompt)
    print(f"Prompt length: {len(prompt)}, Latency: {result['latency']:.2f}s, "
          f"Tokens/s: {result['tokens_per_second']:.2f}")
```

---

### 3.3 阶段 3：SFT 训练流程

#### Step 3.1：数据准备

**数据来源**：
1. 使用项目自带的 SFT 数据导出功能
2. 使用公开的 SFT 数据集
3. 自己构造数据

```python
# 使用项目的数据导出功能
python scripts/build_sft.py \
    --source smolagents/ml-intern-sessions \
    --target your-username/ml-intern-sft \
    --days 7

# 或者使用公开数据集
from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train[:1000]")
```

**数据格式验证**：
```python
# 验证数据格式
def validate_sft_data(dataset):
    required_columns = ["messages"]
    for col in required_columns:
        assert col in dataset.column_names, f"Missing column: {col}"

    # 检查 messages 格式
    sample = dataset[0]
    messages = sample["messages"]
    assert isinstance(messages, list), "messages should be a list"

    for msg in messages:
        assert "role" in msg, "Each message should have 'role'"
        assert "content" in msg, "Each message should have 'content'"
        assert msg["role"] in ["system", "user", "assistant", "tool"]

    print("Data format validation passed!")

validate_sft_data(dataset)
```

#### Step 3.2：训练脚本编写

**基础 SFT 训练脚本**：

```python
#!/usr/bin/env python3
"""SFT Training Script for Qwen2.5-7B"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# 1. 加载模型和分词器
model_name = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="kernels-community/flash-attn2"  # 使用 Hub kernel
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载数据集
dataset = load_dataset("your-username/ml-intern-sft", split="train")

# 3. 训练参数
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    save_strategy="epoch",
    report_to="tensorboard",
    bf16=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
)

# 4. 创建训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    max_seq_length=4096,
)

# 5. 开始训练
trainer.train()

# 6. 保存模型
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
```

#### Step 3.3：分布式训练配置

**使用 DeepSpeed ZeRO-3**：

```json
// ds_config.json
{
    "bf16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

**启动分布式训练**：

```bash
# 使用 torchrun
torchrun --nproc_per_node=8 train_sft.py \
    --deepspeed ds_config.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset your-username/ml-intern-sft \
    --output_dir ./output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4

# 或者使用 accelerate
accelerate launch --num_processes=8 --mixed_precision=bf16 train_sft.py
```

#### Step 3.4：训练监控

**使用 Trackio（项目内置）**：

```python
# 在训练脚本中添加
import trackio

# 初始化
trackio.init(project="ml-intern-sft", name="qwen2.5-7b-sft")

# 记录指标
for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        loss = train_step(batch)
        trackio.log({"loss": loss, "epoch": epoch, "step": step})

# 完成
trackio.finish()
```

**使用 TensorBoard**：

```bash
# 启动 TensorBoard
tensorboard --logdir ./output --port 6006

# 查看训练曲线
# - loss
# - learning_rate
# - grad_norm
```

---

### 3.4 阶段 4：完整流程验证

#### Step 4.1：端到端测试

```python
# 测试完整的 Agent 工作流
def test_end_to_end():
    # 1. 启动 Agent
    agent = MLIntern(model="local/Qwen/Qwen2.5-7B-Instruct")

    # 2. 提交任务
    result = agent.submit("Create a simple Python calculator")

    # 3. 验证结果
    assert result.success
    assert "calculator" in result.output.lower()

    # 4. 检查工具调用
    assert len(result.tool_calls) > 0
    assert any(tc.name == "write" for tc in result.tool_calls)

    print("End-to-end test passed!")

test_end_to_end()
```

#### Step 4.2：性能基准测试

```python
# 性能基准测试
import time
import statistics

def benchmark_agent(tasks, num_runs=5):
    results = []

    for task in tasks:
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            agent.submit(task)
            latencies.append(time.time() - start)

        results.append({
            "task": task,
            "avg_latency": statistics.mean(latencies),
            "p95_latency": sorted(latencies)[int(0.95 * len(latencies))],
            "std_dev": statistics.stdev(latencies)
        })

    return results

tasks = [
    "What is 2+2?",
    "Write a hello world program",
    "Explain machine learning",
    "Create a data processing script"
]

results = benchmark_agent(tasks)
for r in results:
    print(f"Task: {r['task'][:50]}...")
    print(f"  Avg: {r['avg_latency']:.2f}s, P95: {r['p95_latency']:.2f}s")
```

#### Step 4.3：文档整理

**复现文档模板**：

```markdown
# ML Intern 复现报告

## 环境配置
- 硬件：8x RTX 4090, 128GB RAM
- 软件：Python 3.11, PyTorch 2.x, CUDA 12.x

## 复现步骤
1. 基础框架搭建
2. 本地 LLM 集成
3. SFT 训练
4. 完整流程验证

## 遇到的问题及解决方案
1. 问题：xxx
   解决方案：xxx

## 性能基准
- 本地 LLM 推理：xx tokens/s
- SFT 训练：xx samples/s
- 端到端延迟：xx 秒

## 结论
- 成功复现了 xxx 功能
- 性能达到 xxx 水平
- 建议 xxx 优化方向
```

---

## 4. 关键挑战与解决方案

### 4.1 挑战 1：本地 LLM 性能

**问题**：本地 7B 模型性能可能不如 API 模型

**解决方案**：
1. **量化**：使用 GPTQ/AWQ 量化减少显存占用
2. **更大模型**：使用 13B/70B 模型（需要多卡）
3. **混合方案**：简单任务用本地模型，复杂任务用 API

```python
# 量化示例
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-13B-Instruct",
    quantization_config=quantization_config,
    device_map="auto"
)
```

### 4.2 挑战 2：SFT 训练 OOM

**问题**：13B 模型在 4090 上训练可能 OOM

**解决方案**：
1. **减小 batch size**：从 4 减到 1-2
2. **梯度累积**：保持有效 batch size
3. **梯度检查点**：减少显存占用
4. **DeepSpeed ZeRO-3**：分布式训练
5. **LoRA**：参数高效微调

```python
# LoRA 配置
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出：trainable params: 10,000,000 || all params: 7,000,000,000 || trainable%: 0.14%
```

### 4.3 挑战 3：分布式训练通信

**问题**：8 卡训练通信开销大

**解决方案**：
1. **NVLink**：确保卡间有高速互联
2. **优化通信**：使用 NCCL 后端
3. **梯度压缩**：减少通信量

```bash
# 检查 GPU 互联
nvidia-smi topo -m

# 设置 NCCL 环境变量
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
```

### 4.4 挑战 4：数据质量

**问题**：SFT 数据质量影响训练效果

**解决方案**：
1. **数据清洗**：过滤低质量数据
2. **数据增强**：增加多样性
3. **人工审核**：抽样检查

```python
# 数据质量检查
def check_data_quality(dataset):
    issues = []

    for i, sample in enumerate(dataset):
        messages = sample["messages"]

        # 检查空消息
        if not messages:
            issues.append(f"Sample {i}: empty messages")
            continue

        # 检查角色顺序
        expected_roles = ["system", "user", "assistant"]
        actual_roles = [m["role"] for m in messages]
        if actual_roles[0] != "system":
            issues.append(f"Sample {i}: first message should be system")

        # 检查内容长度
        for j, msg in enumerate(messages):
            if len(msg["content"]) > 10000:
                issues.append(f"Sample {i}, message {j}: content too long")

    return issues

issues = check_data_quality(dataset)
print(f"Found {len(issues)} issues")
```

---

## 5. 学习路径建议

### 5.1 第一周：基础理解

**目标**：理解项目架构和核心概念

**学习内容**：
1. 阅读 README.md 和 AGENTS.md
2. 理解 Agent 架构设计
3. 了解工具系统设计

**实践任务**：
1. 搭建基础环境
2. 运行基本对话
3. 测试工具调用

### 5.2 第二周：深入核心

**目标**：理解核心实现细节

**学习内容**：
1. 阅读 agent/core/agent_loop.py
2. 理解上下文管理
3. 了解错误处理机制

**实践任务**：
1. 集成本地 LLM
2. 测试死循环检测
3. 验证上下文压缩

### 5.3 第三周：训练实践

**目标**：掌握 SFT 训练流程

**学习内容**：
1. 阅读 SFT 数据格式
2. 理解训练参数配置
3. 了解分布式训练

**实践任务**：
1. 准备训练数据
2. 编写训练脚本
3. 执行分布式训练

### 5.4 第四周：优化完善

**目标**：优化性能和完善文档

**学习内容**：
1. 性能优化技巧
2. 监控和调试
3. 最佳实践

**实践任务**：
1. 性能基准测试
2. 问题排查和解决
3. 文档整理

---

## 附录：快速参考命令

### 环境配置

```bash
# 克隆项目
git clone https://github.com/huggingface/ml-intern.git
cd ml-intern

# 安装依赖
uv sync
uv tool install -e .

# 配置环境变量
export HF_TOKEN=your_token
export GITHUB_TOKEN=your_token
```

### 本地 LLM 部署

```bash
# vLLM 启动（7B）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 8000

# vLLM 启动（13B，双卡）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-13B-Instruct \
    --tensor-parallel-size 2 \
    --port 8001

# Ollama 启动
ollama serve
ollama pull qwen2.5:7b
```

### SFT 训练

```bash
# 单卡训练
python train_sft.py \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset your-username/ml-intern-sft \
    --output_dir ./output

# 多卡训练
torchrun --nproc_per_node=8 train_sft.py \
    --deepspeed ds_config.json \
    --model_name Qwen/Qwen2.5-7B-Instruct \
    --dataset your-username/ml-intern-sft \
    --output_dir ./output

# LoRA 训练
python train_sft_lora.py \
    --model_name Qwen/Qwen2.5-13B-Instruct \
    --dataset your-username/ml-intern-sft \
    --output_dir ./output_lora
```

### 监控和调试

```bash
# TensorBoard
tensorboard --logdir ./output --port 6006

# GPU 监控
watch -n 1 nvidia-smi

# 日志查看
tail -f /tmp/training.log
```

---

## 总结

**8卡4090复现ML Intern项目的可行性**：

| 组件 | 可行性 | 说明 |
|------|--------|------|
| **Agent 框架** | ✅ 完全可行 | 无需 GPU |
| **本地 LLM 推理** | ✅ 可行 | 7B-13B 模型，单卡或双卡 |
| **SFT 训练** | ✅ 可行 | 7B-13B 模型，4-8 卡 |
| **完整流程** | ✅ 可行 | 需要合理分配资源 |

**建议的资源分配**：
- 2x 4090：本地 LLM 推理
- 4x 4090：SFT 训练
- 2x 4090：备用/测试

**预期时间**：
- 阶段 1：1-2 天
- 阶段 2：2-3 天
- 阶段 3：3-5 天
- 阶段 4：2-3 天
- **总计：8-13 天**

---

*最后更新：2026-06-27*
*基于 8x RTX 4090 硬件环境*
