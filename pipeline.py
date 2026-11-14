# ml_offline/macro_planner/pipeline.py
import cv2
import numpy as np
import torch
from typing import List
from .config import MacroPlannerConfig
from .schemas import PlannedStroke
from .stage1_deconstructor import PhotoRealismDeconstructor
from .stage2_diff_optimizer import DifferentiableBezierLayerOptimizer
from .stage3_sequencer import HybridMacroSequencePlanner

class RealismMacroPlannerPipeline:
    def __init__(self, config: MacroPlannerConfig = MacroPlannerConfig(), device: str = "cuda"):
        self.cfg = config
        self.device = device
        self.deconstructor = PhotoRealismDeconstructor(config, device)
        self.optimizer = DifferentiableBezierLayerOptimizer(config, device)
        self.sequencer = HybridMacroSequencePlanner(config.canvas_width, config.canvas_height)

    def generate_stroke_plan(self, target_rgb_photo: np.ndarray, metric_depth_map: np.ndarray) -> List[PlannedStroke]:
        """
        Full offline decomposition pipeline:
        1. Quantizes real-world pigment palette via LAB K-Means.
        2. Deconstructs photo into Depth-Semantic + Hertzmann Scale Layers.
        3. Optimizes differentiable Bézier splines per layer (Birth of a Painting).
        4. Sequences strokes via Ward Gestalt Clustering + TSP (Sketch & Paint ECCV 2024).
        """
        # 1. Fit K-Means Palette in LAB space
        kmeans, _ = self.deconstructor.extract_palette(target_rgb_photo)

        # 2. Build Semantic + Hertzmann Multi-Scale Pyramid
        layers = self.deconstructor.build_hertzmann_hierarchy(target_rgb_photo, metric_depth_map)

        # 3. Optimize Candidate Strokes Layer-by-Layer
        res = self.cfg.optim_resolution
        running_canvas = torch.ones((3, res, res), dtype=torch.float32, device=self.device)
        all_candidates: List[PlannedStroke] = []

        for layer in layers:
            _, num_strokes, _ = self.cfg.hertzmann_scales[layer.scale_level]
            layer_strokes, running_canvas = self.optimizer.optimize_layer(
                layer=layer,
                current_canvas=running_canvas,
                num_strokes=num_strokes
            )
            
            # Assign K-Means palette cluster IDs to prevent constant brush washing
            rgb_batch = np.array([s.rgb_color for s in layer_strokes], dtype=np.float32)
            lab_batch = cv2.cvtColor((rgb_batch[None, ...] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[0]
            cluster_ids = kmeans.predict(lab_batch.astype(np.float32))
            
            for stroke, cid in zip(layer_strokes, cluster_ids):
                stroke.palette_cluster_id = int(cid)
            
            all_candidates.extend(layer_strokes)

        # 4. Sequence into Human Pedagogical Order (Sketch & Paint ECCV 2024)
        ordered_stroke_queue = self.sequencer.sequence_strokes_for_human(all_candidates)
        return ordered_stroke_queue