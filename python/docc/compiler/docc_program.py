from abc import ABC, abstractmethod
from typing import Any, Optional

from docc.sdfg import StructuredSDFG
from docc.compiler.compiled_sdfg import CompiledSDFG


class DoccProgram(ABC):

    def __init__(
        self,
        name: str,
        target: str = "none",
        category: str = "server",
        instrumentation_mode: Optional[str] = None,
        capture_args: Optional[bool] = None,
        remote_tuning: bool = False,
    ):
        self.name = name
        self.target = target
        self.category = category
        self.instrumentation_mode = instrumentation_mode
        self.capture_args = capture_args
        self.remote_tuning = remote_tuning
        self.last_sdfg: Optional[StructuredSDFG] = None
        self.cache: dict = {}

    @abstractmethod
    def __call__(self, *args: Any) -> Any:
        pass

    @abstractmethod
    def compile(self, *args: Any, output_folder: Optional[str] = None) -> CompiledSDFG:
        pass

    @abstractmethod
    def to_sdfg(self, *args: Any) -> StructuredSDFG:
        pass

    @abstractmethod
    def _convert_inputs(self, args: tuple) -> tuple:
        pass

    @abstractmethod
    def _convert_outputs(self, result: Any, original_args: tuple) -> Any:
        pass

    def _get_cache_key(self, *args: Any) -> str:
        return ""
