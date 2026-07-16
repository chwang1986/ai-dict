"""
AI 词典 - 种子数据初始化脚本
预填 20 个 AI 常见词条，涵盖 4 个分类。
"""

import os
import sqlite3
from datetime import datetime

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict.db')

# 种子数据
TERMS = [
    # ===== 模型架构 =====
    (
        1, 'Transformer', '变换器', '模型架构',
        """## 概述
Transformer 是一种基于自注意力机制（Self-Attention）的深度学习架构，由 Google 团队在 2017 年的论文《Attention Is All You Need》中提出。它彻底改变了自然语言处理领域，并成为大语言模型（如 GPT、BERT）的基石。

与传统的循环神经网络不同，Transformer 完全摒弃了递归结构，通过并行计算大幅提升了训练效率，同时能够更好地捕捉长距离依赖关系。

## 工作原理
Transformer 的核心是**多头自注意力机制**（Multi-Head Self-Attention）。每个 token 通过三组可学习的线性变换生成 Query、Key、Value 三个向量。注意力权重通过 Query 与 Key 的点积计算得到，再对 Value 进行加权求和，从而让每个位置都能"关注"到序列中的其他位置。

Transformer 还引入了**位置编码**（Positional Encoding）来注入序列顺序信息，因为自注意力本身是位置无关的。整体架构由编码器（Encoder）和解码器（Decoder）堆叠组成，每层包含自注意力子层和前馈神经网络子层，并使用**残差连接**和**层归一化**（Layer Normalization）来稳定训练。

## 应用场景
- **机器翻译**：Transformer 最初就是为翻译任务设计的
- **大语言模型**：GPT 系列使用 Transformer 解码器，BERT 使用编码器
- **多模态模型**：Vision Transformer（ViT）将 Transformer 应用于图像分类

## 示例
```python
# 使用 PyTorch 构建 Transformer 编码器层
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,      # 嵌入维度
    nhead=8,          # 注意力头数
    dim_feedforward=2048
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
```""", '3,15,13'
    ),
    (
        2, 'CNN', '卷积神经网络', '模型架构',
        """## 概述
卷积神经网络（Convolutional Neural Network，CNN）是一种专门用于处理具有网格拓扑结构数据（如图像）的深度学习架构。自 2012 年 AlexNet 在 ImageNet 竞赛中取得突破以来，CNN 一直是计算机视觉任务的主流模型。

CNN 的核心思想是利用**局部感受野**和**权值共享**来大幅减少参数量，同时保留空间层级特征。

## 工作原理
CNN 通过**卷积层**（Convolutional Layer）提取局部特征，卷积核在输入上滑动并计算点积，生成特征图（Feature Map）。随后通过**池化层**（Pooling Layer）进行下采样，减少空间维度并增强平移不变性。多个卷积层和池化层交替堆叠后，特征图被展平并送入全连接层进行分类或回归。

常见的经典架构包括：VGGNet（使用小卷积核堆叠）、ResNet（引入残差连接解决退化问题）、Inception（多尺度并行卷积）等。

## 应用场景
- **图像分类与目标检测**：识别图片中的物体类别和位置
- **医学影像分析**：辅助诊断 X 光片、CT 扫描中的病变
- **人脸识别**：从图像中提取人脸特征进行身份验证

## 示例
```python
# 使用 PyTorch 构建简单的 CNN
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```""", '1,3'
    ),
    (
        3, 'RNN', '循环神经网络', '模型架构',
        """## 概述
循环神经网络（Recurrent Neural Network，RNN）是一种专门处理序列数据的神经网络架构。与传统前馈网络不同，RNN 具有"记忆"能力——它在处理每个时间步的输入时，会考虑之前时间步的隐藏状态，从而捕捉序列中的时序依赖关系。

RNN 的变体包括 LSTM（长短期记忆网络）和 GRU（门控循环单元），它们解决了原始 RNN 梯度消失的问题。

## 工作原理
RNN 在每个时间步 t 接收输入 x_t 和上一时刻的隐藏状态 h_{t-1}，通过一个共享的权重矩阵 W 计算当前隐藏状态 h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b)。这个递归过程使得信息可以在时间维度上传递。

然而，原始 RNN 在处理长序列时面临**梯度消失**问题——早期时间步的梯度信号在反向传播中会指数级衰减。LSTM 通过引入**遗忘门、输入门、输出门**三个门控机制来选择性地记忆和遗忘信息，有效缓解了这一问题。

## 应用场景
- **语音识别**：将音频信号序列转换为文字
- **机器翻译**：将源语言序列翻译为目标语言序列
- **时序预测**：股票价格、天气等时间序列的预测

## 示例
```python
# 使用 PyTorch 构建简单 RNN
import torch.nn as nn

rnn = nn.RNN(
    input_size=128,   # 输入特征维度
    hidden_size=256,  # 隐藏状态维度
    num_layers=2,     # RNN 层数
    batch_first=True  # 输入格式 (batch, seq, feature)
)

# 输入: (batch_size=32, seq_len=10, input_size=128)
output, hidden = rnn(torch.randn(32, 10, 128))
```""", '1,2,15'
    ),
    (
        4, 'GAN', '生成对抗网络', '模型架构',
        """## 概述
生成对抗网络（Generative Adversarial Network，GAN）由 Ian Goodfellow 于 2014 年提出，是一种通过对抗训练来生成数据的深度学习框架。GAN 包含两个网络：**生成器**（Generator）和**判别器**（Discriminator），它们在博弈中共同提升——生成器努力产生逼真的数据来"欺骗"判别器，判别器则努力区分真实数据和生成数据。

GAN 在图像生成领域取得了令人瞩目的成果，能够生成高度逼真的照片、艺术作品和视频。

## 工作原理
GAN 的训练过程是一个**极小极大博弈**（Minimax Game）。生成器 G 接收随机噪声 z，输出伪造数据 G(z)；判别器 D 接收真实数据 x 和伪造数据 G(z)，输出其为真实数据的概率。训练目标是使判别器无法区分真假数据，即达到**纳什均衡**。

损失函数为 min_G max_D V(D, G) = E[log D(x)] + E[log(1 - D(G(z)))]。实践中常用 Wasserstein GAN（WGAN）等变体来改善训练稳定性。GAN 的常见挑战包括**模式坍缩**（Mode Collapse），即生成器只产生有限种类的输出。

## 应用场景
- **图像生成**：生成逼真的人脸、风景画等高分辨率图像
- **图像修复与超分辨率**：补全缺失的图像区域或提升低分辨率图像质量
- **数据增强**：为训练数据不足的场景生成合成数据

## 示例
```python
# GAN 的核心训练循环（伪代码）
for epoch in range(num_epochs):
    # 训练判别器
    real_images = next(dataloader)
    fake_images = generator(noise)
    d_loss = discriminator_loss(real_images, fake_images)
    d_optimizer.step()

    # 训练生成器
    fake_images = generator(noise)
    g_loss = generator_loss(fake_images)
    g_optimizer.step()
```""", '5,17'
    ),
    (
        5, 'Diffusion Model', '扩散模型', '模型架构',
        """## 概述
扩散模型（Diffusion Model）是一类基于概率论的生成模型，通过模拟数据的逐步加噪和去噪过程来生成新样本。它在图像生成质量上已超越 GAN，成为 Stable Diffusion、DALL-E 等主流生成式 AI 产品的核心技术。

扩散模型的核心思想来源于非平衡热力学：前向过程逐渐向数据添加高斯噪声直到变为纯噪声，反向过程则学习从噪声中逐步恢复数据。

## 工作原理
扩散模型包含两个过程：

**前向扩散过程**（Forward Process）：在 T 个时间步中，每步向数据 x_0 添加少量高斯噪声，经过 T 步后数据变为近似纯噪声 x_T。这一过程是预定义的，无需学习。

**反向去噪过程**（Reverse Process）：训练一个神经网络（通常是 U-Net）来预测每一步的噪声或直接预测 x_0，从纯噪声 x_T 出发逐步去噪，最终恢复出清晰的数据样本。训练损失是预测噪声与真实噪声之间的均方误差。

为了加速采样，常用的方法包括 DDIM（去噪扩散隐式模型）和 DPM-Solver，可将采样步数从上千步减少到 20-50 步。

## 应用场景
- **文生图**：根据文本描述生成高质量图像（如 Stable Diffusion）
- **图像编辑**：对现有图像进行风格转换、局部修改
- **音频与视频生成**：生成音乐、语音和视频内容

## 示例
```python
# 扩散模型的前向加噪过程（简化）
import torch

def add_noise(x_0, t, total_steps=1000):
    '在时间步 t 向数据添加噪声'
    beta_schedule = torch.linspace(0.0001, 0.02, total_steps)
    alphas = 1 - beta_schedule
    alpha_bars = torch.cumprod(alphas, dim=0)
    noise = torch.randn_like(x_0)
    alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise
    return x_t, noise
```""", '4,17'
    ),

    # ===== 训练技术 =====
    (
        6, 'Fine-tuning', '微调', '训练技术',
        """## 概述
微调（Fine-tuning）是指在预训练模型的基础上，使用特定任务的数据继续训练，使模型适应新任务的技术。它是迁移学习（Transfer Learning）的核心实践，让开发者能够利用大模型已学到的丰富知识，用少量数据和计算资源获得优异的任务表现。

微调的常见策略包括全量微调和参数高效微调（PEFT），后者以 LoRA 为代表。

## 工作原理
预训练模型已经在大规模数据上学习了通用的语言表示和知识。微调时，通常会**降低学习率**（相比预训练阶段小 1-2 个数量级），并只在目标任务的数据上训练少量轮次。

全量微调会更新模型的所有参数，计算开销大。参数高效微调（PEFT）方法只训练少量额外参数：如 LoRA 冻结原始权重，只训练低秩分解矩阵；Adapter 在 Transformer 层中插入小型适配器模块；Prefix Tuning 只训练前缀向量。

## 应用场景
- **文本分类**：将通用语言模型微调为情感分析、意图识别等分类器
- **领域适配**：让通用模型适应医疗、法律等专业领域
- **指令遵循**：通过指令微调让模型学会按照用户指令行动

## 示例
```python
# 使用 Hugging Face 进行微调（伪代码）
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-chinese", num_labels=2
)

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    num_train_epochs=3,
    learning_rate=2e-5
)
trainer.train()
```""", '7,8,1'
    ),
    (
        7, 'RLHF', '基于人类反馈的强化学习', '训练技术',
        """## 概述
RLHF（Reinforcement Learning from Human Feedback）是一种利用人类偏好反馈来对齐大语言模型行为的技术。它是 ChatGPT 成功的关键因素之一，通过将人类的评价标准融入训练过程，使模型的输出更符合人类期望——不仅要有用，还要安全、诚实、无害。

RLHF 的核心流程分为三步：监督微调、奖励模型训练、强化学习优化。

## 工作原理
RLHF 的训练分为三个阶段：

1. **监督微调**（SFT）：先收集人类标注的高质量问答对，微调预训练模型使其学会基本的对话格式和质量标准。

2. **奖励模型训练**（RM）：让模型对同一问题生成多个回答，由人类标注员对这些回答进行排序。用这些偏好数据训练一个奖励模型，使其能给回答打分（分数越高表示越符合人类偏好）。

3. **PPO 强化学习优化**：使用 PPO（Proximal Policy Optimization）算法，以奖励模型的打分为反馈信号，进一步优化语言模型的策略。在优化过程中加入 KL 散度惩罚，防止模型偏离原始分布太远。

## 应用场景
- **对话式 AI**：让 ChatGPT 等聊天机器人更"像人"地对话
- **安全对齐**：减少模型的有害输出和偏见
- **个性化定制**：根据特定用户群体的偏好调整模型行为

## 示例
```
# RLHF 训练流程示意
预训练模型 → 监督微调(SFT) → 生成多个回答 → 人类排序 → 训练奖励模型(RM)
→ PPO 优化 → 对齐后的模型
```""", '6,9,1'
    ),
    (
        8, 'LoRA', '低秩自适应', '训练技术',
        """## 概述
LoRA（Low-Rank Adaptation）是一种参数高效的微调方法，由微软团队在 2021 年提出。它通过在预训练模型的权重矩阵旁注入可训练的低秩分解矩阵，实现了以极少量参数（通常不到原模型的 1%）达到接近全量微调的效果。

LoRA 大幅降低了微调的显存需求和存储成本，使得在消费级 GPU 上微调大语言模型成为可能。

## 工作原理
LoRA 的核心思想是：微调引起的权重变化矩阵具有**低秩特性**（Low-Rank），即可以用远小于原始维度的矩阵来近似表示。

具体实现中，对于原始权重矩阵 W_0 (d x k)，LoRA 冻结 W_0 不变，旁路添加两个小矩阵 A (d x r) 和 B (r x k)，其中 r << min(d, k) 为秩（通常取 4-64）。前向传播时，输出为 y = W_0 * x + B * A * x。训练时只更新 A 和 B 的参数，大幅减少显存占用。

推理时可以将 LoRA 权重合并回原始权重（W = W_0 + BA），不增加任何推理延迟。多个 LoRA 适配器可以共享一个基础模型，按需切换。

## 应用场景
- **大模型微调**：在有限 GPU 资源上微调 7B-70B 参数的模型
- **多任务适配**：为不同任务训练独立的 LoRA 权重，共享基础模型
- **快速实验**：低成本尝试不同的微调策略和超参数

## 示例
```python
# 使用 PEFT 库应用 LoRA
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

base_model = AutoModelForCausalLM.from_pretrained("llama-7b")

lora_config = LoraConfig(
    r=16,                # 秩
    lora_alpha=32,       # 缩放因子
    target_modules=["q_proj", "v_proj"],  # 应用的层
    lora_dropout=0.05
)

model = get_peft_model(base_model, lora_config)
# 可训练参数仅占原模型的 ~0.1%
model.print_trainable_parameters()
```""", '6,1'
    ),
    (
        9, 'Prompt Engineering', '提示词工程', '训练技术',
        """## 概述
提示词工程（Prompt Engineering）是指通过设计和优化输入给大语言模型的文本提示（Prompt），来引导模型产生更准确、更有用、更符合预期的输出。它是一种无需修改模型参数就能提升模型表现的技术，被称为与大模型"沟通的艺术"。

好的提示词能够显著提升模型在各种任务上的表现，从简单问答到复杂的推理和创作。

## 工作原理
提示词工程的核心在于理解大语言模型如何理解和响应输入。模型基于训练数据中的模式来生成回复，因此提示词的质量直接影响输出质量。

常见的提示技术包括：

- **零样本提示**（Zero-shot）：直接描述任务，不提供示例
- **少样本提示**（Few-shot）：在提示中提供几个输入-输出示例
- **思维链**（Chain-of-Thought）：引导模型逐步推理，如"请一步一步地思考"
- **角色扮演**：设定模型角色，如"你是一位资深的 Python 工程师"
- **结构化提示**：使用分隔符、编号列表等格式清晰组织输入

## 应用场景
- **提升回答质量**：通过精心设计的提示获得更准确、更详细的回答
- **约束输出格式**：指定 JSON、Markdown 等输出格式
- **复杂任务分解**：将多步骤任务拆分为清晰的子任务指令

## 示例
```
# 简单提示
"请解释什么是 Transformer"

# 思维链提示
"请一步一步地解释 Transformer 的自注意力机制是如何工作的，
先解释 Query、Key、Value 的含义，再说明注意力权重的计算过程。"

# 角色扮演 + 结构化输出
"你是一位 AI 研究员。请用以下格式总结这篇论文：
- 研究问题
- 核心方法
- 主要贡献
- 局限性"
```""", '7,10,14'
    ),

    # ===== 基础概念 =====
    (
        10, 'RAG', '检索增强生成', '应用领域',
        """## 概述
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与大语言模型生成相结合的技术架构。它在模型生成回答之前，先从外部知识库中检索相关文档，将检索结果作为上下文提供给模型，从而生成基于事实的、更准确的回答。

RAG 有效缓解了大语言模型的**幻觉**问题，并使模型能够访问最新的、私有的知识。

## 工作原理
RAG 的工作流程分为三个主要阶段：

1. **索引阶段**（Indexing）：将文档切分为较小的文本块（Chunk），使用嵌入模型将每个文本块转换为向量表示，存储到向量数据库中。

2. **检索阶段**（Retrieval）：用户提问时，将问题转换为向量，在向量数据库中进行相似度搜索（通常使用余弦相似度），找到最相关的 k 个文本块。

3. **生成阶段**（Generation）：将用户的原始问题和检索到的相关文本块拼接为提示词，输入给大语言模型，由模型基于检索到的上下文生成最终回答。

## 应用场景
- **企业知识库问答**：基于公司内部文档自动回答员工问题
- **智能客服**：结合产品文档提供准确的客户支持
- **学术研究助手**：基于论文数据库回答研究相关问题

## 示例
```python
# RAG 的简化流程（伪代码）
from vector_db import search

question = "什么是注意力机制？"
# 检索相关文档
results = search(query=question, top_k=3)
# 构建提示词
context = chr(10).join(results)
prompt = "根据以下参考资料回答问题：\n" + context + "\n问题：" + question
# 调用大语言模型生成回答
answer = llm.generate(prompt)
```""", '19,11,9'
    ),
    (
        11, 'Agent', '智能体', '应用领域',
        """## 概述
AI 智能体（Agent）是指能够自主感知环境、做出决策并执行动作以完成特定目标的 AI 系统。与大语言模型单纯生成文本不同，Agent 能够使用工具（如搜索、代码执行、API 调用）、进行规划、从反馈中学习，展现出更强的自主性和实用性。

Agent 是当前 AI 应用的前沿方向，被视为从"对话式 AI"迈向"自主 AI"的关键一步。

## 工作原理
Agent 的核心架构通常包含以下组件：

- **大语言模型作为"大脑"**：负责理解任务、推理决策和生成行动方案
- **工具调用**（Tool Use）：Agent 可以调用外部工具——搜索引擎、计算器、代码解释器、数据库查询等
- **记忆系统**（Memory）：短期记忆保存当前对话上下文，长期记忆存储过往经验和知识
- **规划能力**（Planning）：将复杂任务分解为子任务，制定执行计划
- **行动执行**（Action）：将决策转化为具体的操作并执行

典型的 Agent 循环是：感知输入 → 思考推理 → 选择工具 → 执行动作 → 观察结果 → 继续循环直到任务完成（ReAct 模式）。

## 应用场景
- **自动化办公**：自动处理邮件、日程安排、文档整理
- **代码开发助手**：自主编写、调试和部署代码
- **数据分析**：自动收集数据、生成报告和可视化

## 示例
```
# Agent 的工作循环示例
用户：帮我查一下今天的天气并推荐合适的穿搭

Agent 思考：需要获取天气信息 → 调用天气工具
Agent 行动：调用 get_weather(location="北京")
工具返回：北京今天 25°C，晴
Agent 思考：25°C 晴天，适合穿轻薄衣物
Agent 回复：北京今天 25°C，天气晴朗。建议穿短袖或轻薄衬衫，
          搭配长裤即可，可以带一件薄外套以备早晚温差。
```""", '10,20,12'
    ),
    (
        12, 'MCP', '模型上下文协议', '应用领域',
        """## 概述
MCP（Model Context Protocol，模型上下文协议）是一种开放标准协议，旨在标准化 AI 模型与外部数据源和工具之间的交互方式。由 Anthropic 提出，MCP 定义了一套统一的接口，让 AI 应用能够以一致的方式连接各种数据源和服务，无需为每个集成单独开发适配器。

MCP 类似于 AI 领域的"USB 接口"，提供了一种通用的连接标准。

## 工作原理
MCP 采用**客户端-服务器**架构：

- **MCP 服务器**（Server）：封装了对特定数据源或工具的访问，暴露标准化的资源（Resources）、工具（Tools）和提示（Prompts）端点。例如，一个文件系统 MCP 服务器可以提供文件读取、搜索等功能。

- **MCP 客户端**（Client）：集成在 AI 应用（如 IDE、聊天工具）中，负责与一个或多个 MCP 服务器建立连接，并将服务器提供的功能和数据呈现给 AI 模型。

- **协议通信**：客户端和服务器之间通过 JSON-RPC 2.0 消息格式通信，支持请求-响应和通知两种消息类型。

这种解耦设计使得添加新的数据源只需部署对应的 MCP 服务器，无需修改 AI 应用本身。

## 应用场景
- **IDE 集成**：让 AI 助手访问代码仓库、数据库和文档
- **企业数据连接**：将 AI 模型连接到企业内部的各类系统
- **工具生态**：构建可复用、可组合的 AI 工具市场

## 示例
```json
// MCP 工具调用示例
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "read_file",
    "arguments": {
      "path": "/src/app.py"
    }
  }
}
```""", '11,20'
    ),
    (
        13, 'Embedding', '嵌入/词向量', '基础概念',
        """## 概述
嵌入（Embedding）是将离散的符号（如单词、句子、图像）映射到连续的低维向量空间中的技术。这些向量捕获了对象的语义信息——语义相近的词在向量空间中的距离也相近。嵌入是现代 AI 系统的基石，广泛应用于搜索、推荐、分类等任务。

Word2Vec、GloVe 是经典的词嵌入方法，现代大语言模型则通过上下文动态生成嵌入。

## 工作原理
嵌入的本质是学习一个从高维离散空间到低维连续空间的映射函数。以词嵌入为例，每个词被表示为一个稠密向量（通常 128-1536 维）。

**静态嵌入**（如 Word2Vec）：每个词对应一个固定向量，通过预测上下文（CBOW）或预测中心词（Skip-gram）来训练。著名特性是可以通过向量运算类比关系，如 king - man + woman ≈ queen。

**动态嵌入**（如 BERT）：同一词在不同上下文中会有不同的向量表示，解决了多义词问题。例如"苹果"在水果语境和公司语境下会产生不同的嵌入向量。

嵌入的质量通常通过**相似度任务**、**类比任务**和**下游任务性能**来评估。

## 应用场景
- **语义搜索**：将查询和文档转换为向量，通过相似度匹配实现语义级搜索
- **推荐系统**：将用户和物品嵌入到同一空间，计算相似度进行推荐
- **RAG 系统**：将文本块嵌入并存储到向量数据库中供检索

## 示例
```python
# 使用 Sentence Transformers 生成文本嵌入
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# 编码句子为向量
embeddings = model.encode([
    "人工智能正在改变世界",
    "机器学习是AI的子领域",
    "今天天气真好"
])

# 计算两句的余弦相似度
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"相似度: {sim[0][0]:.4f}")  # 较高，因为都关于AI
```""", '14,19,15'
    ),
    (
        14, 'Token', '词元', '基础概念',
        """## 概述
词元（Token）是自然语言处理模型处理文本的基本单位。在将文本输入模型之前，需要先将文本切分为一系列 token。Token 可以是一个词、一个词的一部分（子词）、甚至一个字符。现代大语言模型普遍采用**子词分词**（Subword Tokenization）策略，在词粒度和字符粒度之间取得平衡。

常见的分词器包括 BPE（Byte Pair Encoding）、WordPiece 和 SentencePiece。

## 工作原理
子词分词的核心思想是：高频词保持为完整的 token，低频词被拆分为更小的子词 token。这样既能覆盖所有词汇（包括未登录词 OOV），又不会让词表过大。

以 BPE 为例，训练过程如下：初始时每个字符都是一个 token，然后统计相邻 token 对的频率，将最高频的对合并为新的 token，重复此过程直到达到预设的词表大小。例如 "lower" 可能被合并为单个 token，而 "lowest" 可能被拆分为 "low" + "est"。

分词结果直接影响模型处理文本的效率。中文文本通常每个字就是一个 token，有时常见词组也会被合并为一个 token。不同模型的分词方式不同，同一段文本在不同模型中可能产生不同数量的 token。

## 应用场景
- **模型输入预处理**：所有文本必须先经过分词才能输入模型
- **Token 计费**：大多数 AI API 按输入和输出的 token 数量计费
- **上下文窗口管理**：模型的上下文长度限制以 token 数为单位（如 128K tokens）

## 示例
```python
# 使用 tiktoken 计算 token 数量
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
text = "人工智能正在改变世界"
tokens = enc.encode(text)
print(f"Token 数量: {len(tokens)}")
print(f"Token 列表: {tokens}")
print(f"还原文本: {enc.decode(tokens)}")
```""", '13,16'
    ),
    (
        15, 'Attention', '注意力机制', '基础概念',
        """## 概述
注意力机制（Attention Mechanism）是深度学习中的一种技术，它允许模型在处理输入时动态地"关注"输入的不同部分，赋予不同位置不同的重要性权重。注意力机制是 Transformer 架构的核心组件，也被广泛应用于计算机视觉、语音识别等领域。

注意力机制解决了序列建模中长距离依赖信息难以传递的问题，是现代 AI 最重要的基础技术之一。

## 工作原理
注意力的计算过程可以概括为三个步骤：

1. **线性变换**：将输入分别映射为 Query（查询）、Key（键）、Value（值）三个向量。Query 代表"我在找什么"，Key 代表"我能提供什么"，Value 是实际的内容。

2. **注意力权重计算**：通过计算 Query 与所有 Key 的相似度（通常用点积），再经过 Softmax 归一化，得到每个位置的注意力权重（0-1 之间，总和为 1）。

3. **加权求和**：用注意力权重对 Value 进行加权求和，得到最终的输出。注意力权重高的位置对输出的贡献更大。

多头注意力（Multi-Head Attention）将上述过程并行执行多次，每次使用不同的线性投影，使模型能同时关注不同类型的信息。

## 应用场景
- **机器翻译**：让解码器关注源句子中与当前翻译词最相关的部分
- **文本摘要**：识别输入文本中最重要的句子和词
- **视觉 Transformer**：让图像模型关注图像中最相关的区域

## 示例
```python
# 简化的自注意力实现
import torch
import torch.nn.functional as F

def self_attention(x):
    'x: (batch, seq_len, d_model)'
    d_k = x.size(-1)
    # 生成 Q, K, V（实际中用不同的线性层）
    Q = K = V = x
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    # Softmax 归一化
    weights = F.softmax(scores, dim=-1)
    # 加权求和
    output = torch.matmul(weights, V)
    return output, weights
```""", '1,3,13'
    ),
    (
        16, 'Inference', '推理', '基础概念',
        """## 概述
推理（Inference）在 AI 领域有两个含义：在机器学习中指模型训练完成后，用新数据进行预测的过程（也叫推断或预测）；在逻辑和认知科学中指从已知前提得出结论的过程。在当前大语言模型的语境下，"推理"通常指模型接收输入并生成输出的整个过程。

推理性能（速度和吞吐量）是部署 AI 模型时的关键考量因素。

## 工作原理
大语言模型的推理过程是一个**自回归**（Autoregressive）的过程：模型每次生成一个 token，将这个 token 拼接到输入中，再生成下一个 token，如此反复直到生成结束标记或达到最大长度。

推理的优化技术包括：

- **KV Cache**：缓存注意力计算中的 Key 和 Value 向量，避免重复计算之前的 token
- **量化**（Quantization）：将模型权重从 FP16 压缩为 INT8/INT4，减少显存占用和计算量
- **批处理**（Batching）：将多个请求合并处理，提高 GPU 利用率
- **投机解码**（Speculative Decoding）：用小模型快速生成候选 token，大模型验证

## 应用场景
- **在线服务**：为聊天机器人、搜索引擎等提供实时响应
- **边缘部署**：在手机、IoT 设备等资源受限的环境中运行模型
- **批量处理**：离线对大量文本进行分类、摘要等处理

## 示例
```python
# 简单的推理循环
def generate(model, prompt, max_tokens=100, temperature=0.7):
    input_ids = tokenize(prompt)
    for _ in range(max_tokens):
        # 前向传播
        output = model(input_ids)
        # 取最后一个 token 的 logits
        logits = output[:, -1, :] / temperature
        # 采样
        next_token = sample(logits)
        input_ids = append(input_ids, next_token)
        if next_token == EOS_TOKEN:
            break
    return decode(input_ids)
```""", '14,17,18'
    ),
    (
        17, 'Temperature', '温度参数', '基础概念',
        """## 概述
温度参数（Temperature）是控制大语言模型生成文本"随机性"的超参数，应用于 Softmax 之前的 logits 上。温度值越高，生成结果越随机、越有创造性；温度值越低，生成结果越确定、越保守。温度是调节模型输出风格的最常用手段之一。

温度参数通常取值范围为 0.0 到 2.0，默认值通常为 1.0。

## 工作原理
模型生成下一个 token 时，先计算每个候选 token 的得分（logits），然后除以温度参数 T，再经过 Softmax 转换为概率分布，最后从该分布中采样。

数学表达为：P(x_i) = softmax(logits_i / T)

- **T = 0**（或接近 0）：相当于总是选择概率最高的 token（贪心解码/Greedy Decoding），输出最确定但可能重复
- **T = 1**：使用模型原始的概率分布
- **T > 1**：概率分布变得更平坦，低概率 token 也有更多机会被选中，输出更多样化
- **0 < T < 1**：概率分布变得更尖锐，高概率 token 更容易被选中，输出更集中

## 应用场景
- **创意写作**：使用较高温度（0.8-1.2）生成更丰富多样的内容
- **代码生成**：使用较低温度（0.0-0.3）确保生成的代码准确可靠
- **数据分析**：使用低温度确保回答的精确性

## 示例
```python
import torch
import torch.nn.functional as F

def sample_with_temperature(logits, temperature=1.0):
    '使用温度参数进行采样'
    # 温度缩放
    scaled_logits = logits / temperature
    # Softmax 转为概率
    probabilities = F.softmax(scaled_logits, dim=-1)
    # 从概率分布中采样
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token

# 不同温度的效果
logits = torch.tensor([1.0, 2.0, 3.0, 0.5])
print("T=0.1:", sample_with_temperature(logits, 0.1))  # 大概率选索引2
print("T=1.0:", sample_with_temperature(logits, 1.0))  # 按原始概率采样
print("T=2.0:", sample_with_temperature(logits, 2.0))  # 更均匀的采样
```""", '16,18'
    ),
    (
        18, 'Hallucination', '幻觉', '基础概念',
        """## 概述
AI 幻觉（Hallucination）是指大语言模型生成看似合理但实际上不正确、无根据或与事实不符的内容的现象。这是当前大语言模型面临的最主要挑战之一，在医疗、法律、新闻等对准确性要求极高的领域尤为危险。

幻觉的本质是模型在"自信地胡说"——生成的文本语法通顺、语气肯定，但内容完全错误。

## 工作原理
幻觉产生的原因是多方面的：

1. **训练数据的局限性**：模型从训练数据中学习模式，如果训练数据包含错误信息或对某些话题覆盖不足，模型可能生成错误内容。

2. **自回归生成的本质**：模型逐 token 生成文本，每个 token 的选择基于概率分布，没有"事实核查"机制。一旦开头生成错误，后续内容会基于错误前提继续"合理"生成。

3. **过度泛化**：模型倾向于生成语法流畅但缺乏事实依据的文本，因为训练目标是最小化预测损失而非确保事实正确。

4. **知识截止**：模型的知识停留在训练数据的时间点，对之后发生的事件可能产生错误的"幻觉"。

缓解幻觉的方法包括：RAG（检索增强生成）、RLHF（人类反馈强化学习）、思维链提示以及多模型交叉验证。

## 应用场景
（这是一个需要**避免**的现象，而非应用场景）
- **医疗诊断**：模型可能编造不存在的医学研究或药物
- **法律咨询**：模型可能虚构法律条文或案例
- **新闻生成**：模型可能捏造事件、人物或数据

## 示例
```
# 幻觉示例
用户：爱因斯坦是在哪年获得诺贝尔奖的？

模型（正确）：爱因斯坦于1921年获得诺贝尔物理学奖。

模型（幻觉）：爱因斯坦于1905年获得诺贝尔物理学奖，
表彰他提出了相对论。
（错误：年份错误，且获奖原因是光电效应而非相对论）
```""", '16,17,10'
    ),

    # ===== 应用领域 =====
    (
        19, 'Vector Database', '向量数据库', '应用领域',
        """## 概述
向量数据库是专门用于存储、索引和查询高维向量数据的数据库系统。在 AI 应用中，向量数据库是构建 RAG（检索增强生成）系统、语义搜索和推荐系统的核心基础设施。它能够高效地在百万甚至亿级向量中找到与查询向量最相似的顶部结果。

主流的向量数据库包括 Milvus、Pinecone、Weaviate、Chroma 和 Qdrant 等。

## 工作原理
向量数据库的核心操作是**近似最近邻搜索**（Approximate Nearest Neighbor, ANN）。精确搜索（如暴力遍历）在大规模数据下速度太慢，ANN 算法通过牺牲少量精度换取数量级的速度提升。

常见的 ANN 索引算法包括：

- **HNSW**（分层可导航小世界图）：通过多层图结构实现高效的近似搜索，是当前最流行的算法之一
- **IVF**（倒排文件索引）：将向量空间划分为多个簇，搜索时只在最近的几个簇中查找
- **PQ**（乘积量化）：将高维向量压缩为短编码，减少内存占用

相似度度量通常使用**余弦相似度**（Cosine Similarity）或**欧氏距离**（Euclidean Distance）。

## 应用场景
- **RAG 系统**：存储文档嵌入向量，供检索增强生成使用
- **语义搜索**：根据语义含义而非关键词匹配来搜索内容
- **推荐系统**：基于用户和物品的向量表示进行相似度推荐

## 示例
```python
# 使用 ChromaDB（轻量级向量数据库）
import chromadb
from sentence_transformers import SentenceTransformer

# 初始化客户端和集合
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("documents")

# 生成嵌入并存储
model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["机器学习是人工智能的子领域", "深度学习使用神经网络"]
embeddings = model.encode(docs).tolist()

collection.add(
    documents=docs,
    embeddings=embeddings,
    ids=["doc1", "doc2"]
)

# 相似度搜索
query_embedding = model.encode(["什么是ML"]).tolist()
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)
```""", '13,10'
    ),
    (
        20, 'Multi-Agent System', '多智能体系统', '应用领域',
        """## 概述
多智能体系统（Multi-Agent System，MAS）是由多个 AI 智能体协作或竞争来完成复杂任务的系统。每个 Agent 可以拥有不同的角色、能力和目标，通过相互通信和协调来解决问题。多智能体系统是 Agent 技术的自然扩展，能够处理单个 Agent 难以独立完成的复杂任务。

代表框架包括 AutoGen（微软）、CrewAI、MetaGPT 等。

## 工作原理
多智能体系统的关键设计要素包括：

1. **角色定义**：为每个 Agent 分配明确的角色和职责，如"研究员"负责收集信息、"评论员"负责评审、"作家"负责撰写。

2. **通信协议**：定义 Agent 之间的消息传递格式和规则。常见模式包括：顺序传递（A → B → C）、广播（A 同时发给 B 和 C）、辩论（多轮讨论）。

3. **协调机制**：确保 Agent 之间有效协作，避免冲突和重复工作。可以设置一个"协调者"Agent 来分配任务和管理流程。

4. **共享记忆**：Agent 之间共享上下文和工作成果，确保信息一致性。可以是共享的黑板（Blackboard）或消息历史。

典型的工作流是：用户提出需求 → 规划 Agent 分解任务 → 各专业 Agent 执行子任务 → 结果汇总与整合 → 最终输出。

## 应用场景
- **软件开发**：产品经理、程序员、测试员等多角色 Agent 协作完成软件项目
- **研究分析**：多个 Agent 分别搜索不同信息源，汇总形成完整报告
- **创意生产**：编剧、导演、美术等多角色 Agent 协作创作内容

## 示例
```
# 多智能体协作示例（伪代码）
team = MultiAgentTeam()

team.add_agent("研究员", role="收集和分析信息")
team.add_agent("作家", role="撰写文章")
team.add_agent("审核员", role="审核质量并给出修改建议")

result = team.run(
    task="写一篇关于 Transformer 架构的科普文章",
    workflow="sequential"  # 研究员 → 作家 → 审核员
)
```""", '11,12'
    ),
]


def init_db():
    """初始化数据库，创建表并插入种子数据。"""
    # 如果数据库已存在，先删除
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 创建词条表
    conn.execute('''
        CREATE TABLE terms (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            title_cn TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            related TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    ''')

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 插入种子数据
    for term in TERMS:
        conn.execute(
            'INSERT INTO terms (id, title, title_cn, category, content, related, created_at, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (term[0], term[1], term[2], term[3], term[4], term[5], now, now)
        )

    conn.commit()
    conn.close()
    print(f'数据库初始化完成，共插入 {len(TERMS)} 个词条。')


if __name__ == '__main__':
    init_db()