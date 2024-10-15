import os
import threading
from typing import Callable, List, Any, Optional
import time
from datetime import datetime as dt


class File:
    def __init__(self, path: str):
        self._path: str = path
        self._creation_time: float = os.path.getmtime(path)

    @property
    def path(self) -> str:
        return self._path

    @property
    def creation_time(self) -> float:
        return self._creation_time

    @property
    def file_name(self):
        return self.path.split("/")[-1]


    @property
    def time_since_creation_str(self) -> str:
        delta = dt.now() - dt.fromtimestamp(self.creation_time)
        if delta.days > 0:
            return f"{delta.days} days ago"
        secs = delta.total_seconds()
        if secs < 61:
            return f"{int(secs)} secs ago"
        mins = int(secs/60)
        if mins < 61:
            return f"{mins} mins ago"
        hours = int(mins / 60)
        if hours < 25:
            return f"{hours} hours ago"

    def __repr__(self):
        return self.file_name


class FileChangeDaemon:
    def __init__(self, callback: Optional[Callable[[List[File]], Any]] = None):
        self._on_change = callback
        self._files: List[File] = []
        self._file_lock = threading.Lock()
        self._path_lock = threading.Lock()
        self._path: str = ""
        self._thread: threading.Thread = threading.Thread(target=self._run, daemon=True)
        self._abort = threading.Event()

    @property
    def path(self) -> str:
        return self._path

    @property
    def files(self) -> List[File]:
        self._file_lock.acquire()
        files = self._files
        self._file_lock.release()
        files.sort(key=lambda file: file.creation_time, reverse=True)
        return files

    def start(self, path_to_observe: str):
        if self._thread.is_alive():
            self.stop()
        if not os.path.exists(path_to_observe) and not os.path.isdir(path_to_observe):
            raise AttributeError(f"Path {path_to_observe} does not exist or is not a directory.")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._path = path_to_observe
        self._abort.clear()
        self._thread.start()

    def stop(self):
        self._abort.set()
        self._thread.join()
        self._path = ""

    def _run(self):
        files = set()
        while not self._abort.is_set():
            new_files = {file for file in os.listdir(self._path) if file.endswith(".pdf")}

            diff = new_files.difference(files)
            if len(diff) != 0 or len(files) != len(new_files):
                self._file_lock.acquire()
                self._files = [File(os.path.join(self._path, file)) for file in new_files]
                self._file_lock.release()
                if self._on_change is not None:
                    self._on_change(self._files)
                files = new_files
            time.sleep(.5)

        if self._on_change is not None:
            self._file_lock.acquire()
            self._files = []
            self._file_lock.release()
            self._on_change([])


if __name__ == '__main__':
    daemon = FileChangeDaemon(print)
    daemon.start("/Users/nicolai/Dev/score_scan_ocr/score_scan_ocr/samples")
    while True:
        pass
