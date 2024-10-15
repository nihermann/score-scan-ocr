from typing import List, Callable

from application_model.deamons import FileChangeDaemon, File


class Model:
    def __init__(self):
        self.file_daemon = FileChangeDaemon(self._on_files_update)
        self._on_files_update_callback: Callable[[List[File]], None] = None

    def try_change_src_dir(self, new_path: str) -> bool:
        try:
            self.file_daemon.start(new_path)
        except AttributeError:
            return False
        return True

    def set_on_files_update_callback(self, callback: Callable[[List[File]], None]):
        self._on_files_update_callback = callback

    def _on_files_update(self, files: List[File]) -> None:
        files.sort(key=lambda file: file.creation_time)

        if self._on_files_update_callback is not None:
            self._on_files_update_callback(files)
