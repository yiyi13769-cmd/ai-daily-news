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
    # 🌍 海外一手（套利源）——国内通常晚1-3天，主动挖掘
    {"name": "Product Hunt AI",   "url": "https://www.producthunt.com/topics/artificial-intelligence/feed", "category": "海外一手"},
    {"name": "Ben's Bites",       "url": "https://bensbites.beehiiv.com/feed",                             "category": "海外一手"},
    {"name": "Hacker News",       "url": "https://news.ycombinator.com/rss",                               "category": "海外一手"},

    # ⚡ AI 工具与实践（可直接用）
    {"name": "Simon Willison",    "url": "https://simonwillison.net/atom/everything/",                     "category": "AI实践"},
    {"name": "TLDR AI",           "url": "https://tldr.tech/ai/rss",                                       "category": "AI实践"},
    {"name": "The Verge AI",      "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "category": "AI实践"},

    # 🎬 AIGC 行业动态（模型/产品发布，影响创作工具链）
    {"name": "TechCrunch AI",     "url": "https://techcrunch.com/tag/artificial-intelligence/feed/",       "category": "AIGC动态"},
    {"name": "VentureBeat AI",    "url": "https://venturebeat.com/feed/",                                  "category": "AIGC动态"},

    # 🧠 认知成长（思维模型、决策框架）
    {"name": "Farnam Street",     "url": "https://fs.blog/feed/",                                          "category": "认知成长"},
    {"name": "Ness Labs",         "url": "https://nesslabs.com/feed",                                      "category": "认知成长"},

    # 🇨🇳 中文资讯
    {"name": "机器之心",           "url": "https://www.jiqizhixin.com/rss",                                 "category": "中文资讯"},
    {"name": "量子位",             "url": "https://www.qbitai.com/feed",                                    "category": "中文资讯"},
    {"name": "36Kr AI",            "url": "https://36kr.com/feed",                                          "category": "中文资讯"},
    {"name": "少数派",             "url": "https://sspai.com/feed",                                         "category": "实践技巧"},
]

# AI/AIGC 关键词过滤（用于非专属 AI 源）
AI_KEYWORDS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "neural network", "llm", "large language model", "gpt", "claude",
    "gemini", "chatgpt", "generative", "diffusion", "transformer", "aigc",
    "midjourney", "stable diffusion", "openai", "anthropic", "deepmind",
    "meta ai", "mistral", "agent", "rag", "fine-tuning", "reinforcement",
    "workflow", "prompt", "automation", "productivity", "tool", "plugin",
    "no-code", "api",
    "人工智能", "机器学习", "大模型", "生成式", "语言模型", "智能体",
    "实践", "工作流", "提效", "自动化", "插件", "技巧", "用法",
    # 认知/成长类
    "mental model", "decision", "thinking", "learning", "habit", "focus",
    "cognitive", "creativity", "framework", "system",
    "思维", "认知", "决策", "成长", "学习方法", "框架",
    # 创作者/自媒体类
    "creator", "content", "audience", "newsletter", "social media",
    "monetize", "growth", "viral", "video", "image generation",
    "创作", "博主", "粉丝", "涨粉", "变现", "选题", "视频",
    # AI Agent 追踪类
    "agent", "autonomous", "agentic", "workflow", "automation",
]

# 需要关键词过滤的非专属 AI 源
NEEDS_FILTER = {
    "Hacker News", "VentureBeat AI", "36Kr AI",
    "Ben's Bites", "Product Hunt AI",
    "Farnam Street", "Ness Labs", "少数派",
}

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
            summary = re.sub(r"<[^>]+>", "", entry.get("summary", ""))[:600].strip()
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
    return unique[:30]


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
            news_text += f"   摘要: {item['summary']}\n"
        news_text += f"   来源: {item['source']} | 链接: {item['link']}\n\n"

    prompt = f"""你是一位服务于 AIGC 自媒体创作者的内容情报官。
读者是做 AIGC 内容的自媒体博主，关注：AI 创作工具、国际信息套利、个人成长认知、提效工作流。

请从以下 {len(news_items)} 条资讯中，按下面的框架整理每日简报：

---

## 🌍 套利先机（1-3条）
国际上已有热度、但国内中文媒体还没充分报道的资讯。
每条格式：
- **[标题/事件]**：1句话说清楚是什么，1句话说"为什么中文创作者现在就该跟进"

## ⚡ AIGC工具速报（最多4条）
直接影响 AI 创作流程的工具更新、新功能、新玩法。
每条格式：
- **[工具名]**：做了什么更新 → 创作者能用它做什么

## 🎯 今日选题推荐（1-2条）
基于今天的资讯，最适合做成短视频/图文/深度文章的方向。
每条格式：
- **选题方向**：[具体标题建议] — 切入角度说明（1句话）

## 🧠 认知升级
（不超过60字）从今天的资讯中提炼一个值得记住的思维框架或认知升级点。

---

筛选原则：
- 优先：工具发布/更新、创作方法论、AI Agent 新进展、认知框架
- 忽略：纯融资新闻、学术论文、基准跑分、政策文件
- 套利判断：资讯来自英文源 且 中文资讯源中没有对应内容 → 标记为套利先机
- **格式要求**：每条引用具体新闻时，末尾附上 `[→ 原文](对应链接url)` Markdown 链接

今日新闻（{today}）：
{news_text}"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=3000,
        temperature=0.7,
    )
    return response.choices[0].message.content


# ── 构建 HTML 邮件 ────────────────────────────────────────────────────────────
def md_to_html(text: str) -> str:
    """把 DeepSeek 返回的 Markdown 转成简单 HTML。"""
    text = html.escape(text)
    # Markdown 链接 [text](url)
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" style="color:#0066cc;text-decoration:none;font-size:12px">\1</a>',
        text
    )
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
    cat_icons = {
        "海外一手": "🌍", "AI实践": "⚡", "AIGC动态": "🎬",
        "认知成长": "🧠", "中文资讯": "🇨🇳", "实践技巧": "🎯",
        "行业动态": "📰", "技术社区": "💬", "技术资讯": "🛠️",
    }
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
    subject = f"🎬 AIGC 每日情报 {today}（{len(news_items)} 条）"
    html_content = build_email_html(summary, news_items)
    send_email(subject, html_content)
    print("🎉 完成！")


if __name__ == "__main__":
    main()
