"""
AI 词典 - Flask 主应用
提供 AI 术语的浏览、搜索、编辑和新增功能。
"""

import os
import re
import random
import sqlite3
import string
from datetime import datetime

import markdown
from flask import Flask, request, render_template, redirect, url_for, g

# ===== 应用配置 =====
app = Flask(__name__)
app.config['DATABASE'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dict.db')


# ===== 数据库工具函数 =====
def get_db():
    """获取数据库连接，复用同一请求中的连接。"""
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception):
    """请求结束时关闭数据库连接。"""
    db = g.pop('db', None)
    if db is not None:
        db.close()


def get_term_count():
    """获取词条总数。"""
    db = get_db()
    row = db.execute('SELECT COUNT(*) AS cnt FROM terms').fetchone()
    return row['cnt'] if row else 0


def render_md(text):
    """将 Markdown 文本渲染为 HTML。"""
    return markdown.markdown(text, extensions=['fenced_code', 'tables'])


def highlight_text(text, keyword):
    """在文本中高亮搜索关键词（用于索引页展示）。"""
    if not keyword:
        return text
    escaped = re.escape(keyword)
    return re.sub(
        f'({escaped})',
        r'<mark>\1</mark>',
        text,
        flags=re.IGNORECASE
    )


def get_all_categories():
    """获取所有分类，用于筛选。"""
    db = get_db()
    rows = db.execute('SELECT DISTINCT category FROM terms ORDER BY category').fetchall()
    return [row['category'] for row in rows]


# ===== 路由 =====

@app.route('/')
def index():
    """索引页：按英文字母顺序展示所有词，支持搜索和分类筛选。"""
    db = get_db()
    q = request.args.get('q', '').strip()
    cat_filter = request.args.get('cat', '').strip()

    # 构建查询
    conditions = []
    params = []

    if q:
        conditions.append('(title LIKE ? OR title_cn LIKE ? OR content LIKE ?)')
        pattern = f'%{q}%'
        params.extend([pattern, pattern, pattern])

    if cat_filter:
        conditions.append('category = ?')
        params.append(cat_filter)

    where_clause = ''
    if conditions:
        where_clause = 'WHERE ' + ' AND '.join(conditions)

    # 按英文字母顺序排序
    query = f'SELECT * FROM terms {where_clause} ORDER BY title COLLATE NOCASE'
    terms = db.execute(query, params).fetchall()

    # 按首字母分组（用于字母索引导航）
    letter_groups = {}
    for letter in string.ascii_uppercase:
        letter_groups[letter] = []

    for term in terms:
        first_char = term['title'][0].upper() if term['title'] else '#'
        if first_char in letter_groups:
            letter_groups[first_char].append(term)
        else:
            # 非字母开头的放入 "#" 组
            if '#' not in letter_groups:
                letter_groups['#'] = []
            letter_groups['#'].append(term)

    # 只保留有词条的字母
    active_letters = [l for l in string.ascii_uppercase if letter_groups[l]]
    if letter_groups.get('#'):
        active_letters.append('#')

    categories = get_all_categories()
    total = get_term_count()

    return render_template(
        'index.html',
        letter_groups=letter_groups,
        active_letters=active_letters,
        categories=categories,
        current_cat=cat_filter,
        q=q,
        total=total,
        highlight=highlight_text if q else None
    )


@app.route('/term/<int:id>')
def term_detail(id):
    """词条详情页：展示 Markdown 渲染后的内容，底部显示相关词链接。"""
    db = get_db()
    term = db.execute('SELECT * FROM terms WHERE id = ?', (id,)).fetchone()
    if term is None:
        return render_template('index.html', letter_groups={}, active_letters=[],
                               categories=[], current_cat='', q='', total=get_term_count()), 404

    # 解析相关词条 ID
    related_ids = []
    if term['related']:
        try:
            related_ids = [int(x.strip()) for x in term['related'].split(',') if x.strip().isdigit()]
        except ValueError:
            pass

    # 查询相关词条
    related_terms = []
    if related_ids:
        placeholders = ','.join('?' * len(related_ids))
        related_terms = db.execute(
            f'SELECT * FROM terms WHERE id IN ({placeholders})',
            related_ids
        ).fetchall()

    # 渲染 Markdown 内容
    content_html = render_md(term['content'])

    total = get_term_count()
    return render_template(
        'term.html',
        term=term,
        content_html=content_html,
        related_terms=related_terms,
        total=total
    )


@app.route('/term/<int:id>/edit', methods=['GET', 'POST'])
def term_edit(id):
    """编辑页：GET 展示表单，POST 保存修改。"""
    db = get_db()
    term = db.execute('SELECT * FROM terms WHERE id = ?', (id,)).fetchone()
    if term is None:
        return redirect(url_for('index'))

    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        title_cn = request.form.get('title_cn', '').strip()
        category = request.form.get('category', '').strip()
        content = request.form.get('content', '')
        related = request.form.get('related', '').strip()

        if not title or not category:
            return render_template(
                'edit.html',
                term=term,
                error='标题和分类为必填字段',
                is_new=False
            )

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        db.execute(
            'UPDATE terms SET title=?, title_cn=?, category=?, content=?, related=?, updated_at=? WHERE id=?',
            (title, title_cn, category, content, related, now, id)
        )
        db.commit()

        return redirect(url_for('term_detail', id=id))

    total = get_term_count()
    return render_template('edit.html', term=term, total=total, error=None, is_new=False)


@app.route('/term/new', methods=['GET', 'POST'])
def term_new():
    """新增词条页：GET 展示空表单，POST 创建新词条。"""
    db = get_db()

    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        title_cn = request.form.get('title_cn', '').strip()
        category = request.form.get('category', '').strip()
        content = request.form.get('content', '')
        related = request.form.get('related', '').strip()

        if not title or not category:
            return render_template(
                'edit.html',
                term=None,
                error='标题和分类为必填字段',
                is_new=True
            )

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor = db.execute(
            'INSERT INTO terms (title, title_cn, category, content, related, created_at, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?)',
            (title, title_cn, category, content, related, now, now)
        )
        db.commit()

        return redirect(url_for('term_detail', id=cursor.lastrowid))

    total = get_term_count()
    return render_template('edit.html', term=None, total=total, error=None, is_new=True)


@app.route('/random')
def random_term():
    """随机跳转到一个词条。"""
    db = get_db()
    row = db.execute('SELECT id FROM terms ORDER BY RANDOM() LIMIT 1').fetchone()
    if row is None:
        return redirect(url_for('index'))
    return redirect(url_for('term_detail', id=row['id']))


# ===== 启动入口 =====
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
