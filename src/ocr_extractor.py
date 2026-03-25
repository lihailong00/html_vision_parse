"""Fast OCR-based extraction with position filtering and VL fallback."""

import re
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image
from loguru import logger

from .inference import InferenceEngine
from config.settings import settings




@dataclass
class TextBlock:
    """Represents a text block detected by OCR."""
    text: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2

    @property
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return 0
        return self.width / self.height


@dataclass
class OCRResult:
    """Result from OCR processing."""
    blocks: List[TextBlock]
    full_text: str
    confidence: float
    is_noise_filtered: bool = False
    method: str = "ocr"


class OCRExtractor:
    """
    Fast OCR-based extractor with position filtering.

    Uses EasyOCR for fast text extraction, applies position-based
    heuristics to filter noise (nav, sidebar, ads, footer), and
    estimates confidence. Falls back to VL model when confidence is low.
    """

    def __init__(self, inference_engine: Optional[InferenceEngine] = None):
        self.engine = inference_engine
        self._ocr_reader = None
        self._use_vl_fallback = inference_engine is not None
        from pathlib import Path
        self._cache_dir = Path.home() / ".easyocr_cache"

    def _get_ocr_reader(self):
        """Lazy load OCR reader."""
        if self._ocr_reader is None:
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(
                    ['en', 'ch_sim'],  # English + Simplified Chinese
                    gpu=True,
                    model_storage_directory=str(self._cache_dir)
                )
                logger.info("easyocr_loaded", languages=['en', 'zh'])
            except ImportError:
                logger.warning("easyocr_not_installed_using_paddle")
                try:
                    from paddleocr import PaddleOCR
                    self._ocr_reader = PaddleOCR(use_angle_cls=True, use_gpu=True, show_log=False)
                    logger.info("paddleocr_loaded")
                except ImportError:
                    logger.error("no_ocr_available_install_easyocr_or_paddleocr")
                    return None
        return self._ocr_reader

    def _extract_text_blocks_ocr(self, image: Image.Image) -> List[TextBlock]:
        """Extract text blocks using OCR."""
        reader = self._get_ocr_reader()
        if reader is None:
            return []

        # Convert PIL Image to numpy array for EasyOCR
        import numpy as np
        img_array = np.array(image.convert('RGB'))

        blocks = []
        try:
            if hasattr(reader, 'readtext'):
                # EasyOCR
                results = reader.readtext(img_array)
                for (bbox, text, conf) in results:
                    if not text.strip():
                        continue
                    x_min = int(min(pt[0] for pt in bbox))
                    y_min = int(min(pt[1] for pt in bbox))
                    x_max = int(max(pt[0] for pt in bbox))
                    y_max = int(max(pt[1] for pt in bbox))
                    blocks.append(TextBlock(
                        text=text.strip(),
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                        confidence=float(conf) if conf else 1.0
                    ))
            else:
                # PaddleOCR
                results = reader.ocr(img_array, cls=True)
                if results and results[0]:
                    for line in results[0]:
                        if not line:
                            continue
                        bbox = line[0]
                        text = line[1][0] if isinstance(line[1], tuple) else line[1]
                        conf = line[1][1] if isinstance(line[1], tuple) else 1.0

                        x_min = int(min(pt[0] for pt in bbox))
                        y_min = int(min(pt[1] for pt in bbox))
                        x_max = int(max(pt[0] for pt in bbox))
                        y_max = int(max(pt[1] for pt in bbox))
                        blocks.append(TextBlock(
                            text=text.strip(),
                            x_min=x_min,
                            y_min=y_min,
                            x_max=x_max,
                            y_max=y_max,
                            confidence=float(conf)
                        ))
        except Exception as e:
            logger.warning("ocr_extraction_failed", error=str(e))

        return blocks

    def _estimate_confidence(
        self,
        blocks: List[TextBlock],
        image_width: int,
        image_height: int,
    ) -> Tuple[float, List[str]]:
        """
        Estimate confidence based on position heuristics.

        Returns:
            Tuple of (confidence_score, list_of_reasons)
        """
        reasons = []
        confidence = 0.8  # Base confidence

        if not blocks:
            return 0.0, ["no_text_detected"]

        # Analyze text distribution
        all_texts = [b.text for b in blocks]
        total_chars = sum(len(t) for t in all_texts)

        # Check 1: Text density in center region (vs edges)
        # Main content typically occupies 30-70% of width, 20-80% of height
        center_blocks = []
        edge_blocks = []

        for block in blocks:
            norm_x = block.center_x / image_width
            norm_y = block.center_y / image_height

            # Center region: 20-80% x, 15-85% y
            if 0.15 < norm_x < 0.85 and 0.12 < norm_y < 0.88:
                center_blocks.append(block)
            else:
                edge_blocks.append(block)

        center_ratio = len(center_blocks) / len(blocks) if blocks else 0
        if center_ratio < 0.5:
            confidence -= 0.2
            reasons.append(f"low_center_ratio_{center_ratio:.2f}")

        # Check 2: Sidebar detection (right side blocks with high y-range)
        right_side_blocks = [b for b in blocks if b.center_x / image_width > 0.85]
        if len(right_side_blocks) > len(blocks) * 0.3:
            confidence -= 0.15
            reasons.append("possible_sidebar_detected")

        # Check 3: Footer detection (blocks in bottom 15%)
        bottom_blocks = [b for b in blocks if b.y_min > image_height * 0.85]
        if len(bottom_blocks) > 10:
            confidence -= 0.1
            reasons.append("possible_footer_detected")

        # Check 4: Navigation detection (blocks in top 8% or forming horizontal lines)
        nav_candidates = [b for b in blocks if b.y_max < image_height * 0.08]
        # If many small blocks in top region, likely nav
        if len(nav_candidates) > 5:
            small_nav = [b for b in nav_candidates if b.height < image_height * 0.02]
            if len(small_nav) > 3:
                confidence -= 0.1
                reasons.append("possible_nav_detected")

        # Check 5: Title detection (large text in top portion)
        title_candidates = [b for b in blocks
                           if b.y_max < image_height * 0.25
                           and len(b.text) > 5
                           and b.height > image_height * 0.03]

        if title_candidates:
            largest_title = max(title_candidates, key=lambda b: b.height)
            if largest_title.height > image_height * 0.05:
                confidence += 0.1
                reasons.append("title_detected")

        # Check 6: Content length sanity
        if total_chars < 50:
            confidence -= 0.2
            reasons.append("very_little_text")
        elif total_chars > 5000:
            confidence += 0.05
            reasons.append("substantial_content")

        # Check 7: Average OCR confidence
        avg_ocr_conf = sum(b.confidence for b in blocks) / len(blocks) if blocks else 0
        if avg_ocr_conf < 0.7:
            confidence -= 0.15
            reasons.append(f"low_ocr_confidence_{avg_ocr_conf:.2f}")

        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))

        return confidence, reasons

    def _filter_noise_blocks(
        self,
        blocks: List[TextBlock],
        image_width: int,
        image_height: int,
    ) -> List[TextBlock]:
        """
        Filter out noise blocks based on position heuristics.

        Rules:
        - Top 8%: likely navigation
        - Bottom 12%: likely footer
        - Right 15% (if wide): likely sidebar
        - Small blocks at edges: likely ads/metadata
        """
        if not blocks:
            return []

        filtered = []

        for block in blocks:
            # Normalize positions
            norm_y_min = block.y_min / image_height
            norm_y_max = block.y_max / image_height
            norm_x_min = block.x_min / image_width
            norm_x_max = block.x_max / image_width

            # Skip top navigation (top 8%)
            if norm_y_max < 0.08 and len(block.text) < 100:
                # But keep if it's a main title
                if block.height > image_height * 0.04:
                    filtered.append(block)
                continue

            # Skip bottom footer (bottom 12%)
            if norm_y_min > 0.88 and len(block.text) < 200:
                continue

            # Skip right sidebar (right 15% for wide images)
            if norm_x_min > 0.85 and image_width > 1000:
                # Unless it's the main content on narrow pages
                continue

            # Skip very small isolated blocks (likely ads, timestamps)
            if block.area < (image_width * image_height * 0.001):
                # But keep if it's part of main content (has nearby blocks)
                nearby = sum(1 for b in blocks
                            if abs(b.center_x - block.center_x) < image_width * 0.2
                            and abs(b.center_y - block.center_y) < image_height * 0.1)
                if nearby < 2:
                    continue

            filtered.append(block)

        return filtered

    def _group_blocks_by_region(
        self,
        blocks: List[TextBlock],
        image_width: int,
        image_height: int,
    ) -> Dict[str, List[TextBlock]]:
        """Group blocks into regions: header, sidebar, content, footer."""
        regions = {
            "header": [],
            "left_sidebar": [],
            "right_sidebar": [],
            "content": [],
            "footer": [],
        }

        for block in blocks:
            norm_y_min = block.y_min / image_height
            norm_y_max = block.y_max / image_height
            norm_x_min = block.x_min / image_width
            norm_x_max = block.x_max / image_width

            if norm_y_max < 0.15:
                regions["header"].append(block)
            elif norm_y_min > 0.85:
                regions["footer"].append(block)
            elif norm_x_max < 0.25 and image_width > 800:
                regions["left_sidebar"].append(block)
            elif norm_x_min > 0.75 and image_width > 800:
                regions["right_sidebar"].append(block)
            else:
                regions["content"].append(block)

        return regions

    def _extract_title_from_blocks(self, blocks: List[TextBlock], image_height: int) -> Optional[str]:
        """Extract likely title from top region blocks."""
        return self.extract_title(blocks, image_height)

    def _extract_time_from_blocks(self, blocks: List[TextBlock]) -> Optional[str]:
        """Extract publish time from blocks using regex."""
        return self.extract_time(blocks)

    def _build_content_from_blocks(self, blocks: List[TextBlock]) -> str:
        """Build clean content string from blocks, sorted by position."""
        return self.build_content(blocks)

    def extract_title(self, blocks: List[TextBlock], image_height: int) -> Optional[str]:
        """Extract likely title from top region blocks (public method)."""
        # Title is usually in top 25%, has largest height, and reasonable length
        title_candidates = [
            b for b in blocks
            if b.y_max < image_height * 0.25
            and len(b.text) >= 3
            and len(b.text) <= 200
            and b.height > image_height * 0.02
        ]

        if not title_candidates:
            return None

        # Pick the one with largest height (likely title)
        title_block = max(title_candidates, key=lambda b: b.height)
        return title_block.text

    def extract_time(self, blocks: List[TextBlock]) -> Optional[str]:
        """Extract publish time from blocks using regex (public method)."""
        time_patterns = [
            # 2026-03-23 14:30
            r"(\d{4}-\d{1,2}-\d{1,2}\s+\d{1,2}:\d{2})",
            # 2026/03/23 14:30
            r"(\d{4}/\d{1,2}/\d{1,2}\s+\d{1,2}:\d{2})",
            # 2026年03月23日 14:30
            r"(\d{4}年\d{1,2}月\d{1,2}日\s*\d{1,2}:\d{2})",
            # 2026-03-23
            r"(\d{4}-\d{1,2}-\d{1,2})",
            # Mar 23, 2026
            r"([A-Za-z]{3}\s+\d{1,2},?\s+\d{4})",
        ]

        for block in blocks:
            text = block.text
            for pattern in time_patterns:
                match = re.search(pattern, text)
                if match:
                    return match.group(1)

        return None

    def build_content(self, blocks: List[TextBlock]) -> str:
        """Build clean content string from blocks, sorted by position."""
        if not blocks:
            return ""

        # Sort by y position first, then x position
        sorted_blocks = sorted(blocks, key=lambda b: (b.y_min, b.x_min))

        # Join with newlines for paragraphs
        lines = []
        prev_y = -1
        prev_x = -1

        for block in sorted_blocks:
            # Add paragraph break if there's a big gap
            if prev_y > 0 and (block.y_min - prev_y) > 30:
                lines.append("")

            # Clean and normalize text
            text = block.text.strip()
            if text:
                lines.append(text)
                prev_y = block.y_max
                prev_x = block.x_max

        return "\n".join(lines)

    def extract_ocr(self, image: Image.Image) -> OCRResult:
        """
        Perform OCR extraction with position filtering.

        Returns:
            OCRResult with filtered text blocks and confidence
        """
        width, height = image.size
        logger.info("ocr_extraction_start", image_size=(width, height))

        start_time = time.perf_counter()

        # Extract text blocks
        blocks = self._extract_text_blocks_ocr(image)

        if not blocks:
            logger.warning("no_text_found_in_image")
            return OCRResult(
                blocks=[],
                full_text="",
                confidence=0.0,
                method="ocr"
            )

        ocr_time = time.perf_counter() - start_time
        logger.info("ocr_blocks_extracted", count=len(blocks), time_ms=ocr_time * 1000)

        # Filter noise
        filtered_blocks = self._filter_noise_blocks(blocks, width, height)

        # Estimate confidence
        confidence, reasons = self._estimate_confidence(filtered_blocks, width, height)

        # Build full text
        full_text = self._build_content_from_blocks(filtered_blocks)

        return OCRResult(
            blocks=filtered_blocks,
            full_text=full_text,
            confidence=confidence,
            is_noise_filtered=len(filtered_blocks) < len(blocks),
            method="ocr"
        )

    def extract_with_fallback(
        self,
        image: Image.Image,
        min_confidence: float = 0.75,
    ) -> Tuple[OCRResult, bool]:
        """
        Extract using OCR, fallback to VL model if confidence is low.

        Args:
            image: PIL Image
            min_confidence: Minimum confidence threshold for OCR success

        Returns:
            Tuple of (OCRResult, used_vl_fallback)
        """
        ocr_result = self.extract_ocr(image)

        if ocr_result.confidence >= min_confidence:
            return ocr_result, False

        if not self._use_vl_fallback or self.engine is None:
            logger.info("low_confidence_but_no_vl_fallback",
                       confidence=ocr_result.confidence,
                       threshold=min_confidence)
            return ocr_result, False

        # Fallback to VL model
        logger.info("triggering_vl_fallback",
                   ocr_confidence=ocr_result.confidence,
                   threshold=min_confidence)

        from .extractor import ContentExtractor, ExtractionResult

        vl_extractor = ContentExtractor(self.engine)
        vl_result = vl_extractor.extract(image)

        # Return a synthetic OCR result indicating VL was used
        if vl_result.parse_error is None:
            return OCRResult(
                blocks=[],
                full_text=vl_result.content or "",
                confidence=vl_result.confidence,
                is_noise_filtered=False,
                method="vl"
            ), True

        return ocr_result, False


def create_hybrid_extractor(
    inference_engine: Optional[InferenceEngine] = None,
    use_ocr_first: bool = True,
) -> Any:
    """
    Factory to create hybrid extractor with optional VL fallback.

    Args:
        inference_engine: VL inference engine for fallback
        use_ocr_first: If True, try OCR first then fallback to VL

    Returns:
        Extractor that chooses method automatically based on confidence
    """
    return OCRExtractor(inference_engine=inference_engine)
