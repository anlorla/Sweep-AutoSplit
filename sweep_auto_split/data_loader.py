"""
LeRobot 2.1 数据加载模块

高效批量读取 LeRobot 数据集，避免逐帧读取的性能问题
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class EpisodeData:
    """单个 Episode 的数据"""
    episode_id: int
    length: int
    fps: int

    # 轨迹数据
    state_trajectory: np.ndarray          # [N, state_dim]
    action_trajectory: np.ndarray         # [N, action_dim]
    ee_pose_trajectory: Optional[np.ndarray] = None  # [N, ee_dim]

    # 时间信息
    timestamps: Optional[np.ndarray] = None
    frame_indices: Optional[np.ndarray] = None

    # 元数据
    task: Optional[str] = None
    task_index: Optional[int] = None

    # 视频路径
    video_paths: Optional[Dict[str, str]] = None


class LeRobotDataLoader:
    """
    LeRobot 2.1 数据集加载器

    支持批量读取 parquet 数据，高效处理大规模数据集
    """

    def __init__(self, dataset_path: str):
        """
        初始化数据加载器

        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        self.metadata: Optional[Dict] = None
        self.episodes_metadata: List[Dict] = []
        self.tasks_metadata: List[Dict] = []

        # 加载元数据
        self._load_metadata()

    def _load_metadata(self):
        """加载数据集元数据"""
        # 加载 info.json
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {info_path}")

        with open(info_path, 'r') as f:
            self.metadata = json.load(f)

        # 加载 episodes.jsonl
        episodes_path = self.dataset_path / "meta" / "episodes.jsonl"
        if episodes_path.exists():
            with open(episodes_path, 'r') as f:
                self.episodes_metadata = [json.loads(line) for line in f]

        # 加载 tasks.jsonl
        tasks_path = self.dataset_path / "meta" / "tasks.jsonl"
        if tasks_path.exists():
            with open(tasks_path, 'r') as f:
                self.tasks_metadata = [json.loads(line) for line in f]

    @property
    def fps(self) -> int:
        """数据集帧率"""
        return self.metadata.get("fps", 10)

    @property
    def total_episodes(self) -> int:
        """总 episode 数量"""
        return self.metadata.get("total_episodes", len(self.episodes_metadata))

    @property
    def total_frames(self) -> int:
        """总帧数"""
        return self.metadata.get("total_frames", 0)

    @property
    def state_dim(self) -> int:
        """状态维度"""
        if "observation.state" in self.metadata.get("features", {}):
            return self.metadata["features"]["observation.state"]["shape"][0]
        return 14  # 默认双臂 7+7

    @property
    def action_dim(self) -> int:
        """动作维度"""
        if "action" in self.metadata.get("features", {}):
            return self.metadata["features"]["action"]["shape"][0]
        return 14

    @property
    def camera_names(self) -> List[str]:
        """相机名称列表"""
        names = []
        for feature_name in self.metadata.get("features", {}):
            if feature_name.startswith("observation.images."):
                name = feature_name.replace("observation.images.", "")
                names.append(name)
        return names

    def get_episode_list(self) -> List[int]:
        """获取所有 episode ID 列表"""
        return [ep["episode_index"] for ep in self.episodes_metadata]

    def get_episode_length(self, episode_id: int) -> int:
        """获取指定 episode 的长度"""
        for ep in self.episodes_metadata:
            if ep["episode_index"] == episode_id:
                return ep["length"]
        raise ValueError(f"Episode {episode_id} not found")

    def get_episode_task(self, episode_id: int) -> Optional[str]:
        """获取指定 episode 的任务描述"""
        for ep in self.episodes_metadata:
            if ep["episode_index"] == episode_id:
                tasks = ep.get("tasks", [])
                return tasks[0] if tasks else None
        return None

    def _get_parquet_path(self, episode_id: int) -> Path:
        """获取 parquet 文件路径"""
        chunks_size = self.metadata.get("chunks_size", 1000)
        episode_chunk = episode_id // chunks_size

        # 使用 data_path 模板
        data_path_template = self.metadata.get(
            "data_path",
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        )

        parquet_path = self.dataset_path / data_path_template.format(
            episode_chunk=episode_chunk,
            episode_index=episode_id
        )

        return parquet_path

    def _get_video_paths(self, episode_id: int) -> Dict[str, str]:
        """获取视频文件路径"""
        chunks_size = self.metadata.get("chunks_size", 1000)
        episode_chunk = episode_id // chunks_size

        video_path_template = self.metadata.get(
            "video_path",
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )

        video_paths = {}
        for camera_name in self.camera_names:
            video_key = f"observation.images.{camera_name}"
            video_path = self.dataset_path / video_path_template.format(
                episode_chunk=episode_chunk,
                video_key=video_key,
                episode_index=episode_id
            )
            if video_path.exists():
                video_paths[camera_name] = str(video_path)

        return video_paths

    def load_episode(self, episode_id: int) -> EpisodeData:
        """
        加载单个 episode 的所有数据

        Args:
            episode_id: Episode ID

        Returns:
            EpisodeData 对象
        """
        parquet_path = self._get_parquet_path(episode_id)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        # 读取 parquet
        df = pd.read_parquet(parquet_path)

        # 提取状态轨迹
        if "observation.state" in df.columns:
            state_trajectory = np.stack(df["observation.state"].values)
        else:
            raise ValueError(f"observation.state not found in episode {episode_id}")

        # 提取动作轨迹
        if "action" in df.columns:
            action_trajectory = np.stack(df["action"].values)
        else:
            action_trajectory = np.zeros((len(df), self.action_dim))

        # 提取末端位姿（可选）
        ee_pose_trajectory = None
        if "observation.ee_pose" in df.columns:
            ee_pose_trajectory = np.stack(df["observation.ee_pose"].values)

        # 提取时间信息
        timestamps = df["timestamp"].values if "timestamp" in df.columns else None
        frame_indices = df["frame_index"].values if "frame_index" in df.columns else None

        # 获取任务信息
        task = self.get_episode_task(episode_id)
        task_index = df["task_index"].iloc[0] if "task_index" in df.columns else None

        # 获取视频路径
        video_paths = self._get_video_paths(episode_id)

        return EpisodeData(
            episode_id=episode_id,
            length=len(df),
            fps=self.fps,
            state_trajectory=state_trajectory,
            action_trajectory=action_trajectory,
            ee_pose_trajectory=ee_pose_trajectory,
            timestamps=timestamps,
            frame_indices=frame_indices,
            task=task,
            task_index=task_index,
            video_paths=video_paths,
        )

    def load_episode_raw(self, episode_id: int) -> pd.DataFrame:
        """
        加载 episode 的原始 DataFrame

        Args:
            episode_id: Episode ID

        Returns:
            pandas DataFrame
        """
        parquet_path = self._get_parquet_path(episode_id)
        return pd.read_parquet(parquet_path)

    def load_multiple_episodes(self, episode_ids: List[int]) -> List[EpisodeData]:
        """
        批量加载多个 episode

        Args:
            episode_ids: Episode ID 列表

        Returns:
            EpisodeData 列表
        """
        return [self.load_episode(ep_id) for ep_id in episode_ids]

    def iter_episodes(self, start: int = 0, end: Optional[int] = None):
        """
        迭代器：逐个返回 episode 数据

        Args:
            start: 起始 episode 索引
            end: 结束 episode 索引（不包含）

        Yields:
            EpisodeData
        """
        if end is None:
            end = self.total_episodes

        episode_ids = self.get_episode_list()[start:end]
        for ep_id in episode_ids:
            yield self.load_episode(ep_id)

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集概要信息"""
        return {
            "path": str(self.dataset_path),
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cameras": self.camera_names,
            "features": list(self.metadata.get("features", {}).keys()),
        }

    def __repr__(self) -> str:
        info = self.get_dataset_info()
        return (
            f"LeRobotDataLoader(\n"
            f"  path={info['path']},\n"
            f"  episodes={info['total_episodes']},\n"
            f"  frames={info['total_frames']},\n"
            f"  fps={info['fps']},\n"
            f"  cameras={info['cameras']}\n"
            f")"
        )
