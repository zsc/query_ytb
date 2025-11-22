#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的 YouTube 元数据爬取脚本：
- 只使用 YouTube Data API v3
- 只抓视频 URL 和 meta 信息，不下载视频
- 支持从文本文件读取多条查询语句
- 自动分页 + 去重 + 导出 CSV

使用方法：
    python youtube_metadata_scraper.py \
        --queries_file queries.txt \
        --output_csv youtube_metadata.csv \
        --max_per_query 200

环境变量：
    需要先设置 YT_API_KEY
"""

import os
import csv
import time
import argparse
import logging
from typing import List, Dict, Set, Optional

import requests

# ===================== 配置区 =====================

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# 每次 search.list 的最大返回数（API 限制 50）
SEARCH_PAGE_SIZE = 50

# 每次 videos.list 最多 50 个 ID
VIDEO_BATCH_SIZE = 50

# 为了避免触发配额/速率限制，可以适当 sleep
REQUEST_SLEEP_SECONDS = 0.1


# ===================== 工具函数 =====================

def get_api_key() -> str:
    api_key = os.environ.get("YT_API_KEY")
    if not api_key:
        raise RuntimeError("环境变量 YT_API_KEY 未设置，请先 export YT_API_KEY=你的key")
    return api_key


def load_queries_from_file(path: str) -> List[str]:
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if q:
                queries.append(q)
    return queries


def youtube_search_query(
    api_key: str,
    query: str,
    max_results: int,
    video_duration: str = "any",
    region_code: Optional[str] = None
) -> List[str]:
    """
    调用 search.list，获取满足 query 的视频 ID 列表。
    - max_results: 这个 query 总共最多获取多少个结果
    - video_duration: "any" / "short" / "medium" / "long"
    - region_code: 如 "US", "JP" 等，选填
    """
    collected_ids: List[str] = []
    next_page_token = None

    logging.info(f"开始搜索: '{query}', 目标条数={max_results}")

    while True:
        if len(collected_ids) >= max_results:
            break

        params = {
            "key": api_key,
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": min(SEARCH_PAGE_SIZE, max_results - len(collected_ids)),
            "videoDuration": video_duration,
            "order": "relevance",
        }
        if region_code:
            params["regionCode"] = region_code
        if next_page_token:
            params["pageToken"] = next_page_token

        resp = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
        if resp.status_code != 200:
            logging.error(f"search.list 请求失败: {resp.status_code} {resp.text}")
            break

        data = resp.json()
        items = data.get("items", [])
        if not items:
            break

        # 收集 videoId
        for item in items:
            video_id = item["id"]["videoId"]
            collected_ids.append(video_id)

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(REQUEST_SLEEP_SECONDS)

    logging.info(f"搜索 '{query}' 完成，获得 {len(collected_ids)} 个 videoId")
    return collected_ids


def youtube_get_videos_details(api_key: str, video_ids: List[str]) -> List[Dict]:
    """
    调用 videos.list，批量获取 video 的 meta 信息。
    返回一个包含字典的列表，每个字典对应一个视频。
    """
    results: List[Dict] = []
    for i in range(0, len(video_ids), VIDEO_BATCH_SIZE):
        batch_ids = video_ids[i:i + VIDEO_BATCH_SIZE]
        params = {
            "key": api_key,
            "part": "snippet,contentDetails,statistics",
            "id": ",".join(batch_ids),
            "maxResults": len(batch_ids),
        }

        resp = requests.get(YOUTUBE_VIDEOS_URL, params=params, timeout=10)
        if resp.status_code != 200:
            logging.error(f"videos.list 请求失败: {resp.status_code} {resp.text}")
            continue

        data = resp.json()
        items = data.get("items", [])
        for item in items:
            results.append(item)

        time.sleep(REQUEST_SLEEP_SECONDS)

    return results


def iso8601_duration_to_seconds(duration: str) -> int:
    """
    将 ISO8601 时长（如 'PT1H2M3S'）转为秒数。
    简易实现，只覆盖 PT..H..M..S 三种单位，够用。
    """
    if not duration.startswith("PT"):
        return 0
    duration = duration[2:]

    hours = minutes = seconds = 0
    num = ""
    for ch in duration:
        if ch.isdigit():
            num += ch
        else:
            if ch == "H":
                hours = int(num)
            elif ch == "M":
                minutes = int(num)
            elif ch == "S":
                seconds = int(num)
            num = ""
    return hours * 3600 + minutes * 60 + seconds


def flatten_video_item(item: Dict) -> Dict:
    """
    把 videos.list 返回的复 json，抽成一个扁平 dict。
    方便后面写入 CSV。
    """
    video_id = item.get("id", "")
    snippet = item.get("snippet", {}) or {}
    content = item.get("contentDetails", {}) or {}
    stats = item.get("statistics", {}) or {}

    duration_iso = content.get("duration", "")
    duration_seconds = iso8601_duration_to_seconds(duration_iso)

    return {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "title": snippet.get("title", ""),
        "description": snippet.get("description", "").replace("\n", " ").strip(),
        "channel_id": snippet.get("channelId", ""),
        "channel_title": snippet.get("channelTitle", ""),
        "published_at": snippet.get("publishedAt", ""),
        "duration_iso8601": duration_iso,
        "duration_seconds": duration_seconds,
        "view_count": stats.get("viewCount", ""),
        "like_count": stats.get("likeCount", ""),
        "comment_count": stats.get("commentCount", ""),
        "tags": "|".join(snippet.get("tags", [])),
        "definition": content.get("definition", ""),
        "caption": content.get("caption", ""),
    }


def write_to_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        logging.warning("没有数据需要写入 CSV。")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    logging.info(f"已写入 CSV：{path}，共 {len(rows)} 行。")


# ===================== 主流程 =====================

def main():
    parser = argparse.ArgumentParser(description="YouTube 元数据爬取脚本")
    parser.add_argument(
        "--queries_file",
        type=str,
        required=True,
        help="包含搜索 query 的文本文件，每行一个"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="youtube_metadata.csv",
        help="输出 CSV 文件路径"
    )
    parser.add_argument(
        "--max_per_query",
        type=int,
        default=200,
        help="每个 query 最多抓取多少个视频（默认 200）"
    )
    parser.add_argument(
        "--video_duration",
        type=str,
        default="long",
        choices=["any", "short", "medium", "long"],
        help="过滤视频时长（short<4min, medium 4-20min, long>20min）"
    )
    parser.add_argument(
        "--region_code",
        type=str,
        default=None,
        help="可选：区域代码，如 US, JP, GB，用于限制搜索区域"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="日志等级: DEBUG / INFO / WARNING / ERROR"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    api_key = get_api_key()
    queries = load_queries_from_file(args.queries_file)
    logging.info(f"加载 {len(queries)} 个查询语句。")

    all_video_ids: Set[str] = set()
    all_flat_rows: List[Dict] = []

    for q in queries:
        # 1) 调用 search.list 获取 videoId 列表
        video_ids = youtube_search_query(
            api_key=api_key,
            query=q,
            max_results=args.max_per_query,
            video_duration=args.video_duration,
            region_code=args.region_code,
        )

        # 去重：只保留没有出现过的 videoId
        new_ids = [vid for vid in video_ids if vid not in all_video_ids]
        if not new_ids:
            logging.info(f"查询 '{q}' 没有新的 videoId（全被去重了）。")
            continue

        logging.info(f"查询 '{q}' 新增 videoId 数量：{len(new_ids)}")
        all_video_ids.update(new_ids)

        # 2) 调用 videos.list 拉取详细 meta
        items = youtube_get_videos_details(api_key, new_ids)
        logging.info(f"videos.list 返回 {len(items)} 条。")

        # 3) 扁平化存成行
        for item in items:
            row = flatten_video_item(item)
            all_flat_rows.append(row)

    # 4) 写入 CSV
    write_to_csv(args.output_csv, all_flat_rows)
    logging.info("任务完成。")


if __name__ == "__main__":
    main()

