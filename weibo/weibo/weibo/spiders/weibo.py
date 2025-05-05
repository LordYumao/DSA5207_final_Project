# -*- coding: utf-8 -*-
import os
import re
import json
import time
import random
from datetime import datetime
from urllib.parse import unquote

import scrapy
from scrapy import Request, signals
from scrapy.utils.project import get_project_settings
from scrapy.exceptions import CloseSpider
from langdetect import DetectorFactory, detect, LangDetectException

# 语言检测稳定性配置
DetectorFactory.seed = 0
CHINESE_REGEX = re.compile(r'[\u4e00-\u9fa5]')  # 汉字检测


class WeiboEnSpider(scrapy.Spider):
    name = 'weibo_en_crawler'
    allowed_domains = ['weibo.com']
    custom_settings = {
        'DOWNLOAD_DELAY': 2.5,
        'CONCURRENT_REQUESTS': 4,
        'RETRY_TIMES': 3,
        'COOKIES_ENABLED': True,
        'AUTOTHROTTLE_ENABLED': True,
    }

    def __init__(self):
        settings = get_project_settings()
        self.keywords = self.load_keywords(settings.get('KEYWORD_FILE'))
        self.start_date = settings.get('START_DATE', '2024-01-01')
        self.end_date = settings.get('END_DATE', datetime.now().strftime('%Y-%m-%d'))
        self.cookies = self.load_cookies(settings.get('COOKIE_FILE'))
        self.retry_limit = 3

    def load_keywords(self, path):
        """加载关键词文件"""
        if not os.path.exists(path):
            raise CloseSpider(f"关键词文件 {path} 不存在")
        with open(path, 'r', encoding='utf-8') as f:
            return [self.encode_keyword(kw.strip()) for kw in f if kw.strip()]

    def encode_keyword(self, keyword):
        """编码特殊搜索关键词"""
        if keyword.startswith('#') and keyword.endswith('#'):
            return f'%23{keyword[1:-1]}%23'
        return keyword

    def load_cookies(self, path):
        """加载登录Cookie"""
        if not os.path.exists(path):
            self.logger.warning("未提供Cookie文件，部分内容可能无法抓取")
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return {item['name']: item['value'] for item in json.load(f)}

    def start_requests(self):
        """生成初始搜索请求"""
        base_url = 'https://s.weibo.com/weibo'
        date_range = f"custom:{self.start_date}-0:{self.end_date}-0"

        for keyword in self.keywords:
            url = f"{base_url}?q={keyword}&region=custom:400:1000&timescope={date_range}"
            yield Request(
                url=url,
                callback=self.parse_search,
                meta={'keyword': keyword, 'retry_count': 0},
                cookies=self.cookies,
                headers={
                    'Referer': 'https://s.weibo.com/',
                    'User-Agent': random.choice(self.settings.get('USER_AGENTS'))
                }
            )

    def parse_search(self, response):
        """解析搜索结果页"""
        if 'verify' in response.url:
            yield self.handle_antispam(response)
            return

        # 提取英文微博
        for weibo in response.xpath('//div[@class="card-wrap"]'):
            if self.is_english_weibo(weibo):
                yield self.parse_weibo_item(weibo, response.meta['keyword'])
                yield from self.parse_comments(weibo, response.meta['keyword'])

        # 处理分页
        yield from self.handle_pagination(response)

    def parse_comments(self, weibo, keyword):
        """解析单条微博的评论"""
        weibo_id = weibo.xpath('@mid').get()
        comments_url = f'https://api.weibo.com/2/comments/show.json?cid={weibo_id}&count=100'  # 示例API

        yield Request(
            url=comments_url,
            callback=self.parse_comments_response,
            meta={'keyword': keyword, 'weibo_id': weibo_id},
            headers={
                'Referer': 'https://s.weibo.com/',
                'User-Agent': random.choice(self.settings.get('USER_AGENTS'))
            }
        )

    def parse_comments_response(self, response):
        """处理评论的响应"""
        data = json.loads(response.text)
        for comment in data.get('comments', []):
            if self.is_english_comment(comment['text']):
                yield {
                    'weibo_id': response.meta['weibo_id'],
                    'comment_id': comment['id'],
                    'user_id': comment['user']['id'],
                    'screen_name': comment['user']['screen_name'],
                    'content': comment['text'],
                    'timestamp': comment['created_at'],
                    'keyword': response.meta['keyword']
                }

    def is_english_comment(self, text):
        """检查评论是否为英文"""
        return detect(text) == 'en'

    def is_english_weibo(self, weibo):
        """双重验证英文内容"""
        try:
            content = weibo.xpath('.//p[@class="txt"]//text()').getall()
            clean_text = self.clean_text(''.join(content))

            # 规则1：排除中文内容
            if CHINESE_REGEX.search(clean_text):
                return False

            # 规则2：语言检测为英文
            return detect(clean_text) == 'en'
        except LangDetectException:
            return False
        except Exception as e:
            self.logger.error(f"语言检测失败: {str(e)}")
            return False

    def clean_text(self, text):
        """清洗微博文本"""
        return text.replace('\u200b', '').replace('\ue627', '').strip()

    def parse_weibo_item(self, weibo, keyword):
        """解析单条微博数据"""
        return {
            'weibo_id': weibo.xpath('@mid').get(),
            'user_id': weibo.xpath('.//a[@class="name"]/@href').get().split('/')[-1],
            'screen_name': weibo.xpath('.//a[@class="name"]/text()').get().strip(),
            'content': self.clean_text(''.join(weibo.xpath('.//p[@class="txt"]//text()').getall())),
            'post_time': weibo.xpath('.//div[@class="from"]/a[1]/text()').get().replace(' ', ''),
            'device': weibo.xpath('.//div[@class="from"]/a[2]/text()').get(),
            'interaction': {
                'reposts': weibo.xpath('.//a[@action-type="feed_list_forward"]/text()').re_first(r'\d+') or 0,
                'comments': weibo.xpath('.//a[@action-type="feed_list_comment"]/text()').re_first(r'\d+') or 0,
                'likes': weibo.xpath('.//a[contains(@action-type,"feed_list_like")]/em/text()').get() or 0
            },
            'keyword': keyword,
            'timestamp': datetime.now().isoformat()
        }

    def handle_pagination(self, response):
        """处理分页请求"""
        next_page = response.xpath('//a[@class="next"]/@href').get()
        if next_page:
            yield Request(
                url=f"https://s.weibo.com{next_page}",
                callback=self.parse_search,
                meta=response.meta,
                cookies=self.cookies,
                priority=10
            )

    def handle_antispam(self, response):
        """处理反爬验证"""
        retry_count = response.meta.get('retry_count', 0)
        if retry_count >= self.retry_limit:
            raise CloseSpider("触发反爬验证，停止爬取")

        self.logger.warning(f"遇到验证页面，等待30秒后重试... (第{retry_count + 1}次)")
        time.sleep(30)
        response.meta['retry_count'] = retry_count + 1
        return response.request.copy()

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def spider_closed(self, reason):
        """爬虫关闭时执行"""
        self.logger.info(f"爬虫已关闭，原因: {reason}")


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'FEED_FORMAT': 'jsonlines',
        'FEED_URI': 'weibo_en_comments.jsonl',
        'LOG_LEVEL': 'INFO',
        'DOWNLOADER_MIDDLEWARES': {
            'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
            'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': 100,
        }
    })
    process.crawl(WeiboEnSpider)
    process.start()