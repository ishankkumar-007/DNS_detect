"""
DNS Spoofing Detection Package
"""

__version__ = "1.0.0"
__author__ = "DNS Security Research Team"

from .preprocessing import DNSDataPreprocessor
from .feature_selection import HybridFeatureSelector
from .model import DNSSpoofingDetector
from .real_time_detection import RealTimeDNSDetector

__all__ = [
    'DNSDataPreprocessor',
    'HybridFeatureSelector',
    'DNSSpoofingDetector',
    'RealTimeDNSDetector'
]
