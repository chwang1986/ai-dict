# AI 词典

一个轻量级的 AI 术语词典 Web 应用，支持词条浏览、搜索、编辑和新增。

## 功能

- 按英文字母 A-Z 顺序浏览所有词条
- 中英文模糊搜索
- 按分类筛选（模型架构、训练技术、基础概念、应用领域）
- 词条详情页（Markdown 渲染）
- 词条编辑（左右分栏，实时预览）
- 新增词条
- 随机跳转
- 相关概念链接

## 技术栈

- 后端：Flask
- 数据库：SQLite
- 前端：HTML + CSS + JavaScript（Jinja2 模板）
- 内容格式：Markdown

## 本地运行

```bash
# 创建虚拟环境并安装依赖
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 初始化数据库（预填 20 个词条）
python3 seed.py

# 启动服务
python3 app.py
```

访问 http://127.0.0.1:5001

## 目录结构

```
ai-dict/
├── app.py              # Flask 主应用
├── seed.py             # 种子数据初始化
├── dict.db             # SQLite 数据库（运行后生成）
├── requirements.txt    # Python 依赖
├── start.sh            # 一键启动脚本
├── templates/          # Jinja2 模板
│   ├── base.html       # 基础模板（导航栏 + 搜索）
│   ├── index.html      # 索引页（字母排序 + 分类筛选）
│   ├── term.html       # 词条详情页
│   └── edit.html       # 编辑/新增页
└── static/
    └── style.css       # 样式文件
```

## 数据库表结构

```sql
CREATE TABLE terms (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,          -- 英文术语
    title_cn    TEXT,                   -- 中文译名
    category    TEXT DEFAULT '未分类',  -- 分类标签
    content     TEXT DEFAULT '',        -- Markdown 正文
    related     TEXT DEFAULT '',        -- 相关词条 ID（逗号分隔）
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP
);
```
