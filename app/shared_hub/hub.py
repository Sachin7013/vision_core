import asyncio
import time
from typing import Dict, List, Optional

from av import VideoFrame


class CameraChannel:
    """
    Per-camera channel that holds the latest frame and broadcasts frames to
    all subscribers via small asyncio queues. Slower subscribers drop old frames
    to keep latency low.
    """

    def __init__(self, camera_id: str, max_sub_queue: int = 3) -> None:
        self.camera_id = camera_id
        self.last_frame: Optional[VideoFrame] = None
        self.last_ts: float = 0.0
        self._subscribers: List[asyncio.Queue] = []
        self._max_sub_queue = max_sub_queue

    def publish(self, frame: VideoFrame) -> None:
        """
        Store the latest frame and broadcast it to all current subscribers.
        If a subscriber's queue is full, drop the oldest frame to make room.
        """
        self.last_frame = frame
        self.last_ts = time.time()

        if not self._subscribers:
            return

        dead_queues: List[asyncio.Queue] = []
        for q in list(self._subscribers):
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(frame)
            except Exception:
                # remove dead/broken queues
                dead_queues.append(q)
        if dead_queues:
            for q in dead_queues:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    def subscribe(self) -> asyncio.Queue:
        """
        Create and return a queue for receiving frames for this camera.
        The latest frame (if available) is pushed immediately.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_sub_queue)
        self._subscribers.append(q)
        if self.last_frame is not None:
            try:
                q.put_nowait(self.last_frame)
            except asyncio.QueueFull:
                pass
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def get_latest(self) -> Optional[VideoFrame]:
        return self.last_frame


class SharedFrameHub:
    """
    Singleton hub that manages per-camera frame channels.

    Usage:
        hub = SharedFrameHub.instance()
        hub.publish(camera_id, frame)
        q = hub.subscribe(camera_id)
        frame = await q.get()
    """

    _instance: Optional["SharedFrameHub"] = None

    def __init__(self) -> None:
        self._channels: Dict[str, CameraChannel] = {}

    @classmethod
    def instance(cls) -> "SharedFrameHub":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _get_channel(self, camera_id: str) -> CameraChannel:
        ch = self._channels.get(camera_id)
        if ch is None:
            ch = CameraChannel(camera_id)
            self._channels[camera_id] = ch
        return ch

    def publish(self, camera_id: str, frame: VideoFrame) -> None:
        self._get_channel(camera_id).publish(frame)

    def subscribe(self, camera_id: str) -> asyncio.Queue:
        return self._get_channel(camera_id).subscribe()

    def unsubscribe(self, camera_id: str, q: asyncio.Queue) -> None:
        self._get_channel(camera_id).unsubscribe(q)

    def get_latest(self, camera_id: str) -> Optional[VideoFrame]:
        return self._get_channel(camera_id).get_latest()
