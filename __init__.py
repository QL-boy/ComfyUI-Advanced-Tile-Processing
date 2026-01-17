from .nodes.tile_splitter import CustomTileSplitter
from .nodes.tile_merger import CustomTileMerger

NODE_CLASS_MAPPINGS = {
    "CustomTileSplitter": CustomTileSplitter,
    "CustomTileMerger": CustomTileMerger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomTileSplitter": "🔧 Advanced Tile Splitter",
    "CustomTileMerger": "🔧 Advanced Tile Merger",
}

# Extend ComfyUI Type Definitions
try:
    from comfy.graph_utils import GraphBuilder
    # Register custom dict type check
    GraphBuilder.add_node_type("TILE_CONFIG", lambda x: isinstance(x, dict) and "splits" in x)
except ImportError:
    pass

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
