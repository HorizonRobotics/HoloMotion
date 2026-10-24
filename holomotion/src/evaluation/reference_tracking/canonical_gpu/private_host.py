"""Private host-call graph for the pinned MuJoCo Warp installation.

Modules stay outside sys.modules. Original functions, globals, caches and Warp
launch entrypoints are never assigned to. Kernel objects retain upstream types.
"""

import functools
import types


class WarpFacade(types.ModuleType):
    def __init__(self, warp, launch, launch_tiled):
        super().__init__("warp")
        self._warp = warp
        self.launch = launch
        self.launch_tiled = launch_tiled

    def __getattr__(self, key):
        try:
            return getattr(self._warp, key)
        except AttributeError:
            from warp._src.context import builtin_functions

            if key in builtin_functions:
                return builtin_functions[key]
            raise


class PrivateHost:
    def __init__(self, warp_facade):
        self.warp = warp_facade
        self.modules = {}
        self.functions = {}

    def module(self, original):
        name = original.__name__
        if not name.startswith("mujoco_warp"):
            return original
        if name in self.modules:
            return self.modules[name]
        private = types.ModuleType(name)
        private.__dict__.update(vars(original))
        self.modules[name] = private
        for key, value in list(vars(original).items()):
            if key == "_KERNEL_CACHE":
                private.__dict__[key] = {}
            elif key == "_STACK":
                private.__dict__[key] = None
            else:
                private.__dict__[key] = self.value(value)
        return private

    def value(self, value):
        if isinstance(value, types.ModuleType):
            if value.__name__ == "warp":
                return self.warp
            return self.module(value)
        if isinstance(value, types.FunctionType) and value.__globals__.get(
            "__name__", ""
        ).startswith("mujoco_warp"):
            return self.function(value)
        if isinstance(
            value, functools._lru_cache_wrapper
        ) and value.__module__.startswith("mujoco_warp"):
            return functools.lru_cache(**value.cache_parameters())(
                self.function(value.__wrapped__)
            )
        return value

    def function(self, original):
        if id(original) in self.functions:
            return self.functions[id(original)]
        import sys

        owner = sys.modules[original.__globals__["__name__"]]
        # Allocate module globals first; cycles resolve through memoized modules.
        private = self.module(owner)
        closure = None
        if original.__closure__:
            closure = tuple(
                types.CellType(self.value(cell.cell_contents))
                for cell in original.__closure__
            )
        result = types.FunctionType(
            original.__code__,
            private.__dict__,
            original.__name__,
            original.__defaults__,
            closure,
        )
        self.functions[id(original)] = result
        result.__kwdefaults__ = original.__kwdefaults__
        result.__annotations__ = original.__annotations__
        result.__dict__.update(original.__dict__)
        result.__module__ = original.__module__
        result.__qualname__ = original.__qualname__
        return result
