#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
youtube_metadata_scraper.py

从一个关键词列表（txt）中，通过 YouTube Data API v3 尽可能多地挖掘频道 ID，
并写出到一个 channel_list.txt（或你指定文件），
在设计上尽量提高「每 100 quota 对应的新频道数」。

核心特性：
- 使用 search.list(type=channel, maxResults=50) 搜索频道
- 对每个关键词自适应决定是否翻页（根据“新频道占比 new_ratio”）
- 全局 quota 预算控制，防止超限
- 去重后的 channelId 输出为 txt
- 可选：对高收益关键词再用 type=video 搜索，从视频结果中挖更多频道（默认关闭）
- 支持 --dry-run 模式，不消耗真实 Quota，仅模拟流程。

依赖：
- Python 3.7+
- requests   （pip install requests）

用法示例：
    export YT_API_KEY="你的 YouTube API Key"
    
    # 真实运行
    python3 youtube_metadata_scraper.py \
        --keywords keywords.txt \
        --output channels.txt \
        --daily-quota 9000
        
    # 模拟运行（测试逻辑，不消耗 API）
    python3 youtube_metadata_scraper.py \
        --keywords keywords.txt \
        --dry-run --log-level DEBUG

"""

import argparse
import logging
import os
import sys
import time
import random
import string
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
SEARCH_QUOTA_COST = 100  # search.list 一次调用固定 100 quota 单位（官方文档）
CHANNELS_LIST_QUOTA_COST = 1  # channels.list 一次调用 1 quota 单位（仅用于可选过滤）


class QuotaManager:
    """简单的配额管理器，在脚本内部追踪剩余 quota。"""

    def __init__(self, daily_quota: int):
        self.daily_quota = daily_quota
        self.remaining = daily_quota

    def consume(self, amount: int) -> bool:
        """尝试消耗 amount 个 quota。成功返回 True，配额不足返回 False。"""
        if self.remaining < amount:
            logging.warning(
                "Quota not enough: need %d, remaining %d", amount, self.remaining
            )
            return False
        self.remaining -= amount
        return True

    def __str__(self) -> str:
        return f"Quota(remaining={self.remaining}, daily={self.daily_quota})"


class YouTubeAPI:
    """YouTube Data API v3 的简单封装。"""

    def __init__(
        self,
        api_key: str,
        quota_manager: QuotaManager,
        http_timeout: int = 10,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
        dry_run: bool = False,
    ):
        self.api_key = api_key
        self.quota = quota_manager
        self.http_timeout = http_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self.dry_run = dry_run

    # ---------- HTTP 基础调用 ----------

    def _request(
        self,
        endpoint: str,
        params: Dict[str, str],
        quota_cost: int,
        desc: str = "",
    ) -> Optional[Dict]:
        """
        通用 GET 请求包装，带简单的重试 & quota 检查。

        :param endpoint: 比如 "search" 或 "videos"
        :param params: query 参数字典
        :param quota_cost: 本次调用的 quota 消耗
        :param desc: 日志描述（可选）
        :return: JSON dict 或 None（出错时）
        """
        # 1. 检查 Quota
        if not self.quota.consume(quota_cost):
            logging.error("Quota exhausted before calling %s", desc or endpoint)
            return None

        # 2. Dry Run 处理
        if self.dry_run:
            logging.info("[DRY-RUN] Would request '%s' (cost %d quota)", desc or endpoint, quota_cost)
            return self._generate_mock_response(endpoint, params)

        # 3. 真实请求逻辑
        url = f"{YOUTUBE_API_BASE}/{endpoint}"
        params = dict(params)
        params["key"] = self.api_key

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, params=params, timeout=self.http_timeout)
                if resp.status_code == 200:
                    return resp.json()

                # 简单的限流/服务器错误处理
                if resp.status_code in (500, 502, 503, 504):
                    logging.warning(
                        "[%s] Server error %s, attempt %d/%d",
                        desc or endpoint,
                        resp.status_code,
                        attempt,
                        self.max_retries,
                    )
                else:
                    logging.error(
                        "[%s] HTTP error %s: %s",
                        desc or endpoint,
                        resp.status_code,
                        resp.text,
                    )
                    break  # 非 5xx，就不要重试了
            except requests.RequestException as e:
                logging.warning(
                    "[%s] Request exception on attempt %d/%d: %s",
                    desc or endpoint,
                    attempt,
                    self.max_retries,
                    e,
                )

            if attempt < self.max_retries:
                time.sleep(self.retry_backoff * attempt)

        # 到这里说明失败了退还本次预扣的 quota
        self.quota.remaining += quota_cost
        logging.error(
            "[%s] Request failed after %d attempts, quota rolled back.",
            desc or endpoint,
            self.max_retries,
        )
        return None

    def _generate_mock_response(self, endpoint: str, params: Dict[str, str]) -> Dict:
        """
        为 Dry Run 模式生成假的 API 响应数据。
        生成随机 ID 以模拟发现了新频道。
        """
        time.sleep(0.1)  # 模拟一点点延迟
        
        if endpoint == "search":
            # 模拟 search.list 响应
            max_results = int(params.get("maxResults", 50))
            search_type = params.get("type", "channel")
            items = []
            
            for _ in range(max_results):
                # 生成一个随机 Channel ID
                rand_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
                fake_channel_id = f"UC_MOCK_{rand_suffix}"
                
                item = {
                    "kind": f"youtube#{search_type}",
                    "etag": "mock_etag",
                    "snippet": {
                        "channelId": fake_channel_id,
                        "title": f"Mock Channel {fake_channel_id}",
                        "description": "This is a mock channel for dry-run."
                    }
                }
                
                # 结构差异：type=channel 时 ID 在 id 对象里，type=video 时 ID 在 snippet 里
                if search_type == "channel":
                    item["id"] = {"kind": "youtube#channel", "channelId": fake_channel_id}
                elif search_type == "video":
                    item["id"] = {"kind": "youtube#video", "videoId": f"VIDEO_{rand_suffix}"}
                
                items.append(item)

            return {
                "kind": "youtube#searchListResponse",
                "etag": "mock_response_etag",
                "nextPageToken": "mock_next_page_token_ABC123", # 总是返回下一页 token 以测试翻页逻辑
                "pageInfo": {"totalResults": 10000, "resultsPerPage": max_results},
                "items": items
            }
            
        elif endpoint == "channels":
            # 模拟 channels.list 响应
            requested_ids = params.get("id", "").split(",")
            items = []
            for cid in requested_ids:
                items.append({
                    "kind": "youtube#channel",
                    "id": cid,
                    "snippet": {"title": "Mock Detail"},
                    "statistics": {"viewCount": "1000", "subscriberCount": "500"}
                })
            return {
                "kind": "youtube#channelListResponse",
                "items": items
            }
            
        return {}

    # ---------- 具体 API ----------

    def search_channels(
        self,
        keyword: str,
        max_results: int = 50,
        page_token: Optional[str] = None,
        order: str = "relevance",
    ) -> Optional[Dict]:
        """
        search.list 搜索频道（type=channel）。
        """
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "channel",
            "maxResults": str(max_results),
            "order": order,
        }
        if page_token:
            params["pageToken"] = page_token

        return self._request(
            "search",
            params=params,
            quota_cost=SEARCH_QUOTA_COST,
            desc=f"search.list (channel) kw='{keyword}'",
        )

    def search_videos(
        self,
        keyword: str,
        max_results: int = 50,
        page_token: Optional[str] = None,
        order: str = "relevance",
    ) -> Optional[Dict]:
        """
        search.list 搜索视频（type=video），可用于扩展挖掘更多频道。
        """
        params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": str(max_results),
            "order": order,
        }
        if page_token:
            params["pageToken"] = page_token

        return self._request(
            "search",
            params=params,
            quota_cost=SEARCH_QUOTA_COST,
            desc=f"search.list (video) kw='{keyword}'",
        )

    def channels_list(
        self,
        channel_ids: List[str],
        parts: str = "snippet,statistics",
    ) -> Optional[Dict]:
        """
        channels.list 查询频道详情（可选，用于质量过滤）。
        一次最多 50 个 ID。
        """
        if not channel_ids:
            return None
        params = {"part": parts, "id": ",".join(channel_ids)}
        return self._request(
            "channels",
            params=params,
            quota_cost=CHANNELS_LIST_QUOTA_COST,
            desc=f"channels.list ({len(channel_ids)} ids)",
        )


class ChannelCollector:
    """
    根据一组关键词，用 YouTubeAPI 尽量挖掘更多频道 ID。
    """

    def __init__(
        self,
        api: YouTubeAPI,
        max_pages_per_keyword: int = 3,
        new_ratio_threshold: float = 0.6,
        use_video_search_for_high_yield_keywords: bool = False,
        high_yield_min_new_ratio: float = 0.8,
    ):
        self.api = api
        self.max_pages_per_keyword = max_pages_per_keyword
        self.new_ratio_threshold = new_ratio_threshold
        self.use_video_search_for_high_yield_keywords = (
            use_video_search_for_high_yield_keywords
        )
        self.high_yield_min_new_ratio = high_yield_min_new_ratio

        self.seen_channels: Set[str] = set()

    # ---------- 工具函数 ----------

    def _extract_channel_ids_from_search_items(
        self, items: List[Dict], source: str
    ) -> Tuple[Set[str], int]:
        """
        从 search.list 返回的 items 中提取 channelId。

        :param items: API 返回的 "items"
        :param source: 日志用字符串，表明来源（channel_search/video_search）
        :return: (新出现的 channelId 集合, 总返回条数)
        """
        new_ids: Set[str] = set()
        total_count = 0

        for item in items:
            total_count += 1
            channel_id = None

            if source == "channel_search":
                # type=channel 时，id 下就是 channelId
                channel_id = (
                    item.get("id", {}).get("channelId")
                    or item.get("snippet", {}).get("channelId")
                )
            elif source == "video_search":
                # type=video 时，snippet 里有 channelId
                channel_id = item.get("snippet", {}).get("channelId")

            if not channel_id:
                continue

            if channel_id not in self.seen_channels:
                new_ids.add(channel_id)

        return new_ids, total_count

    # ---------- 核心逻辑：处理一个关键词 ----------

    def process_keyword(self, keyword: str) -> Dict[str, float]:
        """
        对单个关键词执行：
        - search.list(type=channel) 主流程
        - 自适应翻页
        - （可选）对高收益关键词再执行 search.list(type=video)

        返回一些统计信息，便于日志记录和调优。
        """
        logging.info("Processing keyword: %s", keyword)

        total_new_channels = 0
        total_returned_items = 0
        pages_used = 0

        page_token: Optional[str] = None
        high_yield = False

        # --- 1) 主流程：type=channel ---

        while pages_used < self.max_pages_per_keyword:
            data = self.api.search_channels(
                keyword=keyword, max_results=50, page_token=page_token
            )
            if data is None:
                logging.warning("No data returned for keyword='%s', stop.", keyword)
                break

            items = data.get("items", [])
            if not items:
                logging.info(
                    "No items returned for keyword='%s' (page %d), stop.",
                    keyword,
                    pages_used + 1,
                )
                break

            new_ids, total_count = self._extract_channel_ids_from_search_items(
                items, source="channel_search"
            )

            # 统计
            pages_used += 1
            total_returned_items += total_count

            # 加入全局集合
            self.seen_channels.update(new_ids)
            total_new_channels += len(new_ids)

            # 计算 new_ratio
            new_ratio = len(new_ids) / total_count if total_count > 0 else 0.0

            logging.info(
                "[kw='%s'] page=%d channel_search: total=%d, new=%d, new_ratio=%.2f, global_channels=%d, quota_remaining=%d",
                keyword,
                pages_used,
                total_count,
                len(new_ids),
                new_ratio,
                len(self.seen_channels),
                self.api.quota.remaining,
            )

            # 判断是否高收益关键词（为后续 video_search 做准备）
            # 在 mock 数据中，ratio 通常很高，所以容易触发高收益逻辑
            if new_ratio >= self.high_yield_min_new_ratio and len(new_ids) >= 10:
                high_yield = True

            # 决定是否继续翻页
            next_token = data.get("nextPageToken")
            if (
                not next_token
                or new_ratio < self.new_ratio_threshold
                or self.api.quota.remaining < SEARCH_QUOTA_COST
            ):
                break

            page_token = next_token

        # --- 2) 可选扩展：对高收益关键词用 type=video 再挖一次 ---

        if (
            self.use_video_search_for_high_yield_keywords
            and high_yield
            and self.api.quota.remaining >= SEARCH_QUOTA_COST
        ):
            logging.info(
                "Keyword '%s' is high-yield, performing one video_search to mine extra channels.",
                keyword,
            )
            data_v = self.api.search_videos(keyword=keyword, max_results=50)
            if data_v:
                items_v = data_v.get("items", [])
                if items_v:
                    new_ids_v, total_count_v = (
                        self._extract_channel_ids_from_search_items(
                            items_v, source="video_search"
                        )
                    )
                    # 注意：video_search 可能再次产生“新频道”
                    self.seen_channels.update(new_ids_v)
                    total_new_channels += len(new_ids_v)
                    total_returned_items += total_count_v

                    logging.info(
                        "[kw='%s'] video_search: total=%d, new=%d, global_channels=%d, quota_remaining=%d",
                        keyword,
                        total_count_v,
                        len(new_ids_v),
                        len(self.seen_channels),
                        self.api.quota.remaining,
                    )

        result = {
            "keyword": keyword,
            "pages_used": pages_used,
            "total_returned_items": float(total_returned_items),
            "total_new_channels": float(total_new_channels),
            "avg_new_ratio": (
                (total_new_channels / total_returned_items)
                if total_returned_items > 0
                else 0.0
            ),
        }
        return result


# ---------- 一些辅助函数 ----------


def read_keywords_from_file(path: str) -> List[str]:
    """从 txt 文件读取关键词，一行一个，去除空行，做一次简单去重，保留顺序。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Keywords file not found: {path}")

    seen = set()
    keywords: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            kw = line.strip()
            if not kw:
                continue
            if kw in seen:
                continue
            seen.add(kw)
            keywords.append(kw)

    return keywords


def write_channels_to_file(path: str, channel_ids: Iterable[str]) -> None:
    """将频道 ID 写到 txt 文件中，每行一个。"""
    with open(path, "w", encoding="utf-8") as f:
        for cid in sorted(channel_ids):
            f.write(cid + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape YouTube channel IDs from a keyword list using YouTube Data API v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="YouTube Data API Key。如果省略，将尝试读取环境变量 YT_API_KEY。在 --dry-run 模式下可以是任意字符串。",
    )

    parser.add_argument(
        "--keywords",
        type=str,
        required=True,
        help="关键词列表 txt 文件路径（每行一个关键词）。",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="channel_list.txt",
        help="输频道列表 txt 文件路径（每行一个 channelId）。",
    )

    parser.add_argument(
        "--daily-quota",
        type=int,
        default=9000,
        help="本次脚本可使用的 quota 预算（为了安全，不建议设为 10000 的满值）。",
    )

    parser.add_argument(
        "--max-pages-per-keyword",
        type=int,
        default=20,
        help="每个关键词最多翻几页 search.list(type=channel)。",
    )

    parser.add_argument(
        "--new-ratio-threshold",
        type=float,
        default=0.6,
        help="当某一页的新频道比例低于此阈值时，不再为该关键词继续翻页。",
    )

    parser.add_argument(
        "--use-video-search",
        action="store_true",
        help="对高收益关键词额外执行一次 search.list(type=video)，从视频结果中再挖一些频道。",
    )

    parser.add_argument(
        "--high-yield-min-new-ratio",
        type=float,
        default=0.8,
        help="将关键词视为“收益”的 new_ratio 下限，仅在 --use-video-search 打开时有效。",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="启用干跑模式：不真正调用 API，不消耗真实 Quota，使用模拟数据进行测试。",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志等级。",
    )

    parser.add_argument(
        "--http-timeout",
        type=int,
        default=10,
        help="HTTP 请求超时时间（秒）。",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="请求失败时的最大重试次数。",
    )

    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.5,
        help="重试退避系数（指数型 backoff 的倍数）。",
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 设置日志
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 获取 API Key
    api_key = args.api_key or os.environ.get("YT_API_KEY")
    
    # 如果是 Dry Run，允许没有 API Key
    if args.dry_run:
        logging.info("Running in DRY-RUN mode. No actual API calls will be made.")
        if not api_key:
            api_key = "DUMMY_KEY_FOR_DRY_RUN"
    
    if not api_key:
        logging.error(
            "No API key provided. Use --api-key or set environment variable YT_API_KEY."
        )
        return 1

    # 读取关键词
    try:
        keywords = read_keywords_from_file(args.keywords)
    except Exception as e:
        logging.error("Failed to read keywords file: %s", e)
        return 1

    if not keywords:
        logging.error("No keywords found in file: %s", args.keywords)
        return 1

    logging.info("Loaded %d unique keywords from %s", len(keywords), args.keywords)

    # 初始化 YouTube API 客户端
    quota_manager = QuotaManager(daily_quota=args.daily_quota)
    yt_api = YouTubeAPI(
        api_key=api_key,
        quota_manager=quota_manager,
        http_timeout=args.http_timeout,
        max_retries=args.max_retries,
        retry_backoff=args.retry_backoff,
        dry_run=args.dry_run,
    )

    collector = ChannelCollector(
        api=yt_api,
        max_pages_per_keyword=args.max_pages_per_keyword,
        new_ratio_threshold=args.new_ratio_threshold,
        use_video_search_for_high_yield_keywords=args.use_video_search,
        high_yield_min_new_ratio=args.high_yield_min_new_ratio,
    )

    # 主循环：遍历关键词
    for idx, kw in enumerate(keywords, start=1):
        if quota_manager.remaining < SEARCH_QUOTA_COST:
            logging.warning(
                "Quota almost exhausted, stop processing further keywords. "
                "Processed %d/%d keywords.",
                idx - 1,
                len(keywords),
            )
            break

        logging.info(
            "===== [%d/%d] Keyword: %s | Remaining quota: %d =====",
            idx,
            len(keywords),
            kw,
            quota_manager.remaining,
        )

        stats = collector.process_keyword(kw)
        logging.info(
            "Keyword '%s' stats: pages_used=%d, total_returned_items=%d, "
            "total_new_channels=%d, avg_new_ratio=%.2f",
            stats["keyword"],
            int(stats["pages_used"]),
            int(stats["total_returned_items"]),
            int(stats["total_new_channels"]),
            stats["avg_new_ratio"],
        )

    # 输出频道列表
    write_channels_to_file(args.output, collector.seen_channels)
    logging.info(
        "Finished. Total unique channels: %d. Output written to %s. Quota remaining: %d.",
        len(collector.seen_channels),
        args.output,
        quota_manager.remaining,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
