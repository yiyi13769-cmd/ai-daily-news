#!/usr/bin/env python3
"""AI Daily News Digest - Fetches AI/AIGC news, summarizes with DeepSeek, sends email."""

import feedparser
import requests
import smtplib
import os
import re
import html
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from openai import OpenAI

# ── RSS 源配置 ────────────────────────────────────────────────────────────────
RSS_SOURCES = [
    # 英文 AI 媒体
    {"name": "TechCrunch AI",      "url": "https://techcrunch.com/tag/artificial-intelligence/feed/", "category": "行业动态"},
    {"name": "VentureBeat AI",     "url": "https://venturebeat.com/ai/feed/",                         "category": "行业动态"},
    {"name": "The Verge AI",       "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "category": "行业动态"},
    {"name": "MIT Tech Review",    "url": "https://www.technologyreview.com/feed/",                   "category": "前沿研究"},
    {"name": "Wired AI",           "url": "https://www.wired.com/feed/tag/ai/latest/rss",             "category": "行业动态"},
    {"name": "Ars Technica AI",    "url": "https://feeds.arstechnica.com/arstechnica/technology-lab", "category": "技术资讯"},
    # 学术 / 论文
    {"name": "ArXiv cs.AI",        "url": "https://arxiv.org/rss/cs.AI",                              "category": "前沿研究"},
    {"name": "ArXiv cs.LG",        "url": "https://arxiv.org/rss/cs.LG",                              "category": "前沿研究"},
    {"name": "Papers With Code",   "url": "https://paperswithcode.com/latest.rss",                    "category": "前沿研究"},
    # 技术社区
    {"name": "Hacker News",        "url": "https://news.ycombinator.com/rss",                         "category": "技术社区"},
    {"name": "Reddit ML",          "url": "https://www.reddit.com/r/MachineLearning/.rss",            "category": "技术社区"},
    {"name": "Reddit LocalLLaMA",  "url": "https://www.reddit.com/r/LocalLLaMA/.rss",                "category": "技术社区"},
    # 中文资讯
    {"name": "机器之心",            "url": "https://www.jiqizhixin.com/rss",                           "category": "中文资讯"},
    {"name": "量子位",              "url": "https://www.qbitai.com/feed",                              "category": "中文资讯"},
    {"name": "AI 研习社",           "url": "https://www.yanxishe.com/rss",                             "category": "中文资讯"},
]

# AI/AIGC 关键词过滤（用于非专属 AI 源）
AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "llm", "large language model", "gpt", "claude",
    "gemini", "chatgpt", "generative", "diffusion", "transformer", "aigc",
    "midjourney", "stable diffusion", "openai", "anthropic", "deepmind",
    "meta ai", "mistral", "agent", "rag", "fine-tuning", "reinforcement",
    "人工智能", "机器学习", "大模型", "生成式", "语言模型", "智能体",
]

# 需要关键词过滤的非专属 AI 源
NEEDS_FILTER = {"Hacker News", "MIT Tech Review", "Ars Technica AI", "Reddit ML"}

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AINewsBot/1.0)"}


# ── 抓取新闻 ──────────────────────────────────────────────────────────────────
def fetch_recent_news(hours: int = 24) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    items: list[dict] = []

    for source in RSS_SOURCES:
        try:
            resp = requests.get(source["url"], headers=HEADERS, timeout=15)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"[SKIP] {source['name']}: {e}")
            continue

        for entry in feed.entries[:30]:
            # 解析发布时间
            pub_time = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    pub_time = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    pass

            # 跳过超出时间窗口的条目
            if pub_time and pub_time < cutoff:
                continue

            title   = entry.get("title", "").strip()
            summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:400].strip()
            link    = entry.get("link", "")

            if not title or not link:
                continue

            # 关键词过滤（仅对非专属 AI 源）
            if source["name"] in NEEDS_FILTER:
                combined = (title + " " + summary).lower()
                if not any(kw in combined for kw in AI_KEYWORDS):
                    continue

            items.append({
                "source":   source["name"],
                "category": source["category"],
                "title":    title,
                "summary":  summary,
                "link":     link,
                "pub_time": pub_time,
            })

    # 按标题前 60 字符去重
    seen: set[str] = set()
    unique: list[dict] = []
    for item in items:
        key = item["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)

    # 按时间降序，最多取 20 条
    unique.sort(
        key=lambda x: x["pub_time"] or datetime.min.replace(tzinfo=timezone.utc),
        reverse=True,
    )
    return unique[:20]


# ── DeepSeek 生成摘要 ─────────────────────────────────────────────────────────
def generate_summary(news_items: list[dict]) -> str:
    client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y年%m月%d日")
    news_text = ""
    for i, item in enumerate(news_items, 1):
        news_text += f"{i}. 【{item['category']}】{item['title']}\n"
        if item["summary"]:
            news_text += f"   摘要: {item['summary'][:200]}\n"
        news_text += f"   来源: {item['source']}\n\n"

    prompt = f"""你是专业的 AI 行业分析师，请将以下 {len(news_items)} 条 AI/AIGC 新闻整理成每日简报。

要求：
1. 输出**今日重点**（最重要的 3 条，每条 2 句话）
2. 输出**分类速览**（按行业动态/前沿研究/技术社区/中文资讯分组，每条 1 句话）
3. 最后写**今日总结**（不超过 60 字，点出当天最大趋势）
4. 全部用中文，语言简洁有力，避免废话
5. 使用 Markdown 格式（## 和 ### 标题）

今日新闻（{today}）：
{news_text}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.7,
    )
    return response.choices[0].message.content


# ── 构建 HTML 邮件 ────────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    """把 DeepSeek 返回的 Markdown 转成简单 HTML。"""
    text = html.escape(text)
    # 标题
    text = re.sub(r"^## (.+)$",  r'<h2 style="color:#4a4a8a;margin:20px 0 8px;font-size:18px">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r"^### (.+)$", r'<h3 style="color:#666;margin:14px 0 6px;font-size:15px">\1</h3>',   text, flags=re.MULTILINE)
    # 加粗
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # 列表项（- 或 数字.）
    text = re.sub(r"^[-*] (.+)$",   r'<li style="margin:4px 0">\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\. (.+)$",  r'<li style="margin:4px 0">\1</li>', text, flags=re.MULTILINE)
    # 换行
    text = text.replace("\n", "<br>\n")
    return text


def build_email_html(summary: str, news_items: list[dict]) -> str:
    today = datetime.now(timezone(timedelta(hours=8))).strftime("%Y年%m月%d日 %A")
    summary_html = md_to_html(summary)

    # 按分类整理原文链接
    categories: dict[str, list] = defaultdict(list)
    for item in news_items:
        categories[item["category"]].append(item)

    links_html = ""
    cat_icons = {"行业动态": "📰", "前沿研究": "🔬", "技术社区": "💬", "中文资讯": "🇨🇳", "技术资讯": "🛠️"}
    for cat, items in categories.items():
        icon = cat_icons.get(cat, "📌")
        links_html += f'<h3 style="color:#555;margin:16px 0 8px;font-size:14px;border-bottom:1px solid #eee;padding-bottom:4px">{icon} {html.escape(cat)}</h3>'
        for item in items:
            title  = html.escape(item["title"])
            source = html.escape(item["source"])
            links_html += (
                f'<div style="margin:5px 0;font-size:13px">'
                f'<a href="{item["link"]}" style="color:#0066cc;text-decoration:none">{title}</a> '
                f'<span style="color:#aaa">— {source}</span>'
                f'</div>'
            )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','PingFang SC',sans-serif">
  <div style="max-width:680px;margin:20px auto;background:white;border-radius:16px;overflow:hidden;box-shadow:0 4px 20px rgba(0,0,0,0.08)">

    <!-- Header -->
    <div style="background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:36px 32px;text-align:center">
      <div style="font-size:40px;margin-bottom:8px">🤖</div>
      <h1 style="color:white;margin:0;font-size:26px;font-weight:700;letter-spacing:1px">AI 每日简报</h1>
      <p style="color:rgba(255,255,255,0.8);margin:10px 0 0;font-size:14px">{today} &nbsp;·&nbsp; DeepSeek 智能生成 &nbsp;·&nbsp; {len(news_items)} 条资讯</p>
    </div>

    <!-- AI Summary -->
    <div style="padding:28px 32px;border-bottom:1px solid #f0f0f0">
      <div style="line-height:1.9;color:#333;font-size:14px">
        {summary_html}
      </div>
    </div>

    <!-- Original Links -->
    <div style="padding:24px 32px">
      <h2 style="color:#333;margin:0 0 12px;font-size:16px">📎 原文链接</h2>
      {links_html}
    </div>

    <!-- Footer -->
    <div style="background:#f8f9fa;padding:16px 32px;text-align:center;border-top:1px solid #eee">
      <p style="color:#aaa;font-size:12px;margin:0">
        由 GitHub Actions 自动生成 &nbsp;|&nbsp;
        <a href="https://github.com" style="color:#aaa">取消订阅</a>
      </p>
    </div>
  </div>
</body>
</html>"""


# ── 发送邮件 ──────────────────────────────────────────────────────────────────
def send_email(subject: str, html_content: str) -> None:
    smtp_server = os.environ["SMTP_SERVER"]
    smtp_port   = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user   = os.environ["SMTP_USER"]
    smtp_pass   = os.environ["SMTP_PASS"]
    email_to    = os.environ["EMAIL_TO"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"AI 每日简报 <{smtp_user}>"
    msg["To"]      = email_to
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.ehlo()
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(smtp_user, email_to.split(","), msg.as_string())

    print(f"✅ 邮件已发送至 {email_to}")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("📡 正在抓取新闻...")
    news_items = fetch_recent_news(hours=24)
    print(f"✅ 共获取 {len(news_items)} 条 AI/AIGC 资讯")

    if not news_items:
        print("⚠️  今日无新闻，跳过发送。")
        return

    print("🧠 DeepSeek 生成摘要中...")
    summary = generate_summary(news_items)

    print("📧 构建邮件并发送...")
    today   = datetime.now(timezone(timedelta(hours=8))).strftime("%Y/%m/%d")
    subject = f"🤖 AI 每日简报 {today}（{len(news_items)} 条）"
    html_content = build_email_html(summary, news_items)
    send_email(subject, html_content)
    print("🎉 完成！")


if __name__ == "__main__":
    main()
