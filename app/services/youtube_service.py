import glob
import os
import re
import shutil
import sys
from typing import List, Dict, Optional, Tuple
from loguru import logger
from urllib.parse import urlparse
from uuid import uuid4

from app.utils import utils


class YoutubeService:
    def __init__(self):
        self.supported_formats = ['mp4', 'mkv', 'webm', 'flv', 'avi']
        self.supported_hosts = {
            "youtube.com",
            "www.youtube.com",
            "m.youtube.com",
            "music.youtube.com",
            "youtube-nocookie.com",
            "www.youtube-nocookie.com",
            "youtu.be",
            "www.youtu.be",
        }

    def _get_yt_dlp(self):
        try:
            import yt_dlp
        except ImportError as exc:
            raise RuntimeError("缺少 yt-dlp 依赖，请先安装 requirements.txt 中的依赖") from exc

        return yt_dlp

    def _normalize_youtube_url(self, url: str) -> str:
        """校验并标准化 YouTube URL。"""
        normalized_url = (url or "").strip()
        if not normalized_url:
            raise ValueError("YouTube URL不能为空")

        if "://" not in normalized_url:
            normalized_url = f"https://{normalized_url}"

        parsed = urlparse(normalized_url)
        host = parsed.netloc.lower().split(":")[0]
        if parsed.scheme not in ("http", "https") or not host:
            raise ValueError("请输入有效的 YouTube URL")
        if host not in self.supported_hosts and not host.endswith(".youtube.com"):
            raise ValueError("仅支持 YouTube 视频链接")

        return normalized_url

    def _sanitize_filename_stem(self, name: str) -> str:
        """生成可安全写入本地文件系统的文件名。"""
        safe_name = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", name or "")
        safe_name = re.sub(r"\s+", " ", safe_name).strip(" ._")
        return (safe_name or "youtube_video")[:120]

    def _resolution_height(self, resolution: str) -> Optional[int]:
        match = re.search(r"(\d+)p", resolution or "")
        if not match:
            return None
        return int(match.group(1))

    def _ffmpeg_location(self) -> Optional[str]:
        """yt-dlp can use a bundled venv ffmpeg even when PATH is not activated."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path

        venv_ffmpeg = os.path.join(os.path.dirname(sys.executable), "ffmpeg")
        if os.path.exists(venv_ffmpeg):
            return venv_ffmpeg

        return None

    def _get_video_formats(self, url: str) -> List[Dict]:
        """获取视频可用的格式列表"""
        yt_dlp = self._get_yt_dlp()
        normalized_url = self._normalize_youtube_url(url)
        ydl_opts = {
            'quiet': True,
            'no_warnings': True
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(normalized_url, download=False)
                formats = info.get('formats', [])

                format_list = []
                for f in formats:
                    height = f.get('height')
                    format_info = {
                        'format_id': f.get('format_id', 'N/A'),
                        'ext': f.get('ext', 'N/A'),
                        'resolution': f"{height}p" if height else f.get('format_note', 'N/A'),
                        'filesize': f.get('filesize', 'N/A'),
                        'vcodec': f.get('vcodec', 'N/A'),
                        'acodec': f.get('acodec', 'N/A')
                    }
                    format_list.append(format_info)

                return format_list
        except Exception as e:
            logger.error(f"获取视频格式失败: {str(e)}")
            raise

    def _validate_format(self, output_format: str) -> None:
        """验证输出格式是否支持"""
        if output_format.lower() not in self.supported_formats:
            raise ValueError(
                f"不支持的视频格式: {output_format}。"
                f"支持的格式: {', '.join(self.supported_formats)}"
            )

    def download_video_sync(
            self,
            url: str,
            resolution: str,
            output_format: str = 'mp4',
            rename: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        同步下载指定分辨率的视频。

        Returns:
            Tuple[str, str, str]: (task_id, output_path, filename)
        """
        try:
            yt_dlp = self._get_yt_dlp()
            normalized_url = self._normalize_youtube_url(url)
            task_id = str(uuid4())
            self._validate_format(output_format)

            base_resolution = resolution.split('p')[0] + 'p'
            formats = self._get_video_formats(normalized_url)

            target_format = None
            requested_height = self._resolution_height(base_resolution)
            video_formats = [fmt for fmt in formats if fmt['resolution'] != 'N/A' and fmt['vcodec'] != 'none']
            if requested_height:
                candidates = [
                    fmt for fmt in video_formats
                    if (self._resolution_height(fmt['resolution']) or 0) <= requested_height
                ]
                candidates.sort(
                    key=lambda fmt: (
                        self._resolution_height(fmt['resolution']) or 0,
                        fmt.get('ext') == output_format.lower(),
                    ),
                    reverse=True,
                )
                if candidates:
                    target_format = candidates[0]

            if target_format is None:
                for fmt in video_formats:
                    fmt_resolution = fmt['resolution']
                    fmt_base_resolution = fmt_resolution.split('p')[0] + 'p'
                    if fmt_base_resolution == base_resolution:
                        target_format = fmt
                        break

            if target_format is None:
                available_resolutions = {
                    fmt['resolution'].split('p')[0] + 'p'
                    for fmt in video_formats
                }
                available_text = ', '.join(sorted(available_resolutions)) or "无"
                raise ValueError(
                    f"未找到 {base_resolution} 或更低分辨率的视频。"
                    f"可用分辨率: {available_text}"
                )

            output_dir = utils.video_dir()
            os.makedirs(output_dir, exist_ok=True)

            if rename and rename.strip():
                filename_stem = self._sanitize_filename_stem(rename)
            else:
                preview_opts = {'quiet': True, 'no_warnings': True}
                with yt_dlp.YoutubeDL(preview_opts) as ydl:
                    info = ydl.extract_info(normalized_url, download=False)
                title = self._sanitize_filename_stem(info.get('title', task_id))
                filename_stem = f"{task_id}_{title}"

            output_template = os.path.join(output_dir, f"{filename_stem}.%(ext)s")
            expected_output_path = os.path.join(output_dir, f"{filename_stem}.{output_format.lower()}")

            format_selector = target_format['format_id']
            if target_format.get('acodec') == 'none':
                format_selector = f"{target_format['format_id']}+bestaudio[ext=m4a]/best"

            ydl_opts = {
                'format': format_selector,
                'outtmpl': output_template,
                'merge_output_format': output_format.lower(),
                'quiet': True,
                'no_warnings': True,
                'noprogress': True,
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': output_format.lower(),
                }]
            }
            ffmpeg_location = self._ffmpeg_location()
            if ffmpeg_location:
                ydl_opts['ffmpeg_location'] = ffmpeg_location

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(normalized_url, download=True)

            output_path = expected_output_path
            if not os.path.exists(output_path):
                matches = glob.glob(os.path.join(output_dir, f"{filename_stem}.*"))
                if matches:
                    output_path = max(matches, key=os.path.getmtime)

            filename = os.path.basename(output_path)
            logger.info(f"视频下载成功: {output_path}")
            return task_id, output_path, filename

        except Exception:
            logger.exception("下载视频失败")
            raise

    async def download_video(
            self,
            url: str,
            resolution: str,
            output_format: str = 'mp4',
            rename: Optional[str] = None
    ) -> Tuple[str, str, str]:
        """
        下载指定分辨率的视频

        Args:
            url: YouTube视频URL
            resolution: 目标分辨率 ('2160p', '1440p', '1080p', '720p' etc.)
                       注意：对于类似'1080p60'的输入会被处理为'1080p'
            output_format: 输出视频格式
            rename: 可选的重命名

        Returns:
            Tuple[str, str, str]: (task_id, output_path, filename)
        """
        return self.download_video_sync(url, resolution, output_format, rename)
