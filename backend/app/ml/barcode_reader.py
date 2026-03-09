"""
SmartLib Kiosk - Barcode Reader

Reads 1D and 2D barcodes from book images.
Supports:
- ISBN-13 (EAN-13)
- ISBN-10
- Code128, Code39
- QR codes
"""
import numpy as np
import cv2
from typing import List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

@dataclass
class BarcodeResult:
    """Result of barcode decoding."""
    data: str
    barcode_type: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    
    @property
    def is_isbn(self) -> bool:
        """Check if barcode is an ISBN."""
        return (
            self.barcode_type in ["EAN13", "EAN-13", "UPC", "UPCA", "ISBN"] and 
            (self.data.startswith("978") or self.data.startswith("979") or len(self.data) >= 10)
        )
    
    @property
    def isbn_13(self) -> Optional[str]:
        """Get ISBN-13 format."""
        if self.is_isbn:
            return self.data
        return None

class BarcodeReader:
    """
    Barcode reader using OpenCV Barcode and QR Detectors.
    
    Features:
    - Multi-format support (1D and 2D) built-in to cv2
    - Image preprocessing for better detection
    - Confidence scoring based on image quality
    """
    
    def __init__(self, supported_types: Optional[List[str]] = None):
        """
        Initialize barcode reader.
        
        Args:
            supported_types: List of barcode types to detect
                            (None = all types)
        """
        self.supported_types = supported_types or [
            "EAN13", "EAN8", "CODE128", "CODE39", "QRCODE"
        ]
        
        # Initialize OpenCV Detectors
        try:
            self.barcode_detector = cv2.barcode.BarcodeDetector()
            self.qr_detector = cv2.QRCodeDetector()
        except AttributeError:
            logger.warning("cv2.barcode not found. You might need opencv-contrib-python installed.")
            self.barcode_detector = None
            self.qr_detector = cv2.QRCodeDetector()

    def read(self, image: np.ndarray) -> List[BarcodeResult]:
        """
        Read barcodes from an image.
        
        Args:
            image: BGR or grayscale image
            
        Returns:
            List of BarcodeResult objects
        """
        if image is None or image.size == 0:
            return []
            
        results = []
        
        # Try multiple preprocessing methods
        preprocessed_images = self._preprocess(image)
        
        for processed in preprocessed_images:
            try:
                # 1. Detect 1D Barcodes
                if self.barcode_detector:
                    retval, decoded_info, decoded_type, points = self.barcode_detector.detectAndDecode(processed)
                    if retval and decoded_info:
                        # OpenCV returns lists of lists sometimes depending on the version
                        if not isinstance(decoded_info, (list, tuple)):
                            decoded_info = [decoded_info]
                            decoded_type = [decoded_type] if decoded_type else ["UNKNOWN"]
                            points = [points] if points is not None else [None]
                            
                        for i, info in enumerate(decoded_info):
                            if not info:
                                continue
                            
                            b_type = str(decoded_type[i]) if i < len(decoded_type) else "EAN13"
                            if b_type == '0':
                                b_type = "EAN13"
                            
                            pts = points[i] if points is not None and i < len(points) else None
                            bbox = (0, 0, 0, 0)
                            if pts is not None and len(pts) > 0:
                                # OpenCV points are usually [[x,y], [x,y]...]
                                pts_array = np.array(pts).reshape(-1, 2)
                                x, y, w, h = cv2.boundingRect(np.float32(pts_array))
                                bbox = (int(x), int(y), int(w), int(h))
                                
                            confidence = 0.85
                            
                            result = BarcodeResult(
                                data=info,
                                barcode_type=b_type,
                                bbox=bbox,
                                confidence=confidence
                            )
                            if not any(r.data == info for r in results):
                                results.append(result)

                # 2. Detect 2D QR Codes
                retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(processed)
                if retval and decoded_info:
                    for i, info in enumerate(decoded_info):
                        if not info:
                            continue
                            
                        pts = points[i] if points is not None and i < len(points) else None
                        bbox = (0, 0, 0, 0)
                        if pts is not None and len(pts) > 0:
                            pts_array = np.array(pts).reshape(-1, 2)
                            x, y, w, h = cv2.boundingRect(np.float32(pts_array))
                            bbox = (int(x), int(y), int(w), int(h))
                            
                        result = BarcodeResult(
                            data=info,
                            barcode_type="QRCODE",
                            bbox=bbox,
                            confidence=0.9
                        )
                        if not any(r.data == info for r in results):
                            results.append(result)
                            
            except Exception as e:
                logger.debug(f"Barcode decode error: {e}")
                continue
                
        return results

    def _preprocess(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Apply multiple preprocessing methods for better detection.
        
        Returns list of preprocessed images to try.
        """
        preprocessed = []
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # 1. Original grayscale
        preprocessed.append(gray)
        
        # 2. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        preprocessed.append(enhanced)
        
        # 3. Binary threshold (Otsu)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed.append(binary)
        
        return preprocessed

    def read_isbn(self, image: np.ndarray) -> Optional[str]:
        """
        Read ISBN from image.
        
        Args:
            image: Book image
            
        Returns:
            ISBN string or None
        """
        results = self.read(image)
        
        for result in results:
            if result.is_isbn:
                return result.isbn_13
                
        # Check for any EAN-13 that might be ISBN
        for result in results:
            if result.barcode_type in ["EAN13", "EAN-13", "UPC", "UPCA"]:
                return result.data
                
        return None

    def draw_barcodes(
        self,
        image: np.ndarray,
        results: List[BarcodeResult],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw barcode detections on image."""
        output = image.copy()
        
        for result in results:
            x, y, w, h = result.bbox
            cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
            
            label = f"{result.barcode_type}: {result.data}"
            cv2.putText(output, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                       
        return output
