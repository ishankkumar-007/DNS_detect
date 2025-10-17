"""
Quick Inference Script
Run predictions on new DNS traffic data
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from real_time_detection import RealTimeDNSDetector
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run DNS Spoofing Detection Inference")
    
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained model file (.txt)'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Path to input CSV file with DNS flows'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save predictions CSV (optional)'
    )
    parser.add_argument(
        '--max-flows', type=int, default=None,
        help='Maximum number of flows to process'
    )
    parser.add_argument(
        '--stream', action='store_true',
        help='Process in streaming mode'
    )
    
    return parser.parse_args()


def batch_inference(detector, input_path, output_path=None, max_flows=None):
    """
    Run batch inference on CSV file
    
    Args:
        detector: RealTimeDNSDetector instance
        input_path: Path to input CSV
        output_path: Path to save predictions
        max_flows: Maximum flows to process
    """
    logger.info(f"Loading data from {input_path}")
    
    # Read CSV
    df = pd.read_csv(input_path, low_memory=False)
    
    if max_flows:
        df = df.head(max_flows)
    
    logger.info(f"Processing {len(df)} flows...")
    
    # Convert to list of dictionaries
    flows_data = df.to_dict('records')
    
    # Run predictions
    results = detector.predict_batch(flows_data, measure_latency=True)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df['flow_id'] = df.index
    
    # Combine with original data
    output_df = pd.concat([df, results_df], axis=1)
    
    # Display summary
    logger.info("\n" + "="*60)
    logger.info("INFERENCE RESULTS")
    logger.info("="*60)
    logger.info(f"Total flows processed: {len(results_df)}")
    logger.info(f"Malicious flows detected: {results_df['is_malicious'].sum()}")
    logger.info(f"Detection rate: {results_df['is_malicious'].sum()/len(results_df)*100:.2f}%")
    
    # Display predictions breakdown
    logger.info("\nPrediction Distribution:")
    logger.info(results_df['predicted_label'].value_counts().to_string())
    
    # Performance stats
    perf_stats = detector.get_performance_stats()
    logger.info(f"\nPerformance:")
    logger.info(f"  Average latency: {perf_stats['avg_latency_ms']:.2f}ms")
    logger.info(f"  P95 latency: {perf_stats['p95_latency_ms']:.2f}ms")
    logger.info(f"  Meets <100ms SLA: {perf_stats['meets_sla']}")
    logger.info("="*60)
    
    # Save results
    if output_path:
        output_df.to_csv(output_path, index=False)
        logger.info(f"\nPredictions saved to {output_path}")
    
    return output_df


def stream_inference(detector, input_path, max_flows=None):
    """
    Run streaming inference
    
    Args:
        detector: RealTimeDNSDetector instance
        input_path: Path to input CSV
        max_flows: Maximum flows to process
    """
    from real_time_detection import DNSFlowSimulator
    
    # Create flow generator
    simulator = DNSFlowSimulator(input_path)
    
    # Define callback for malicious detections
    def alert_callback(flow_data, result):
        if result['is_malicious']:
            logger.warning(f"⚠️  ALERT: {result['predicted_label']} detected "
                         f"(confidence: {result['confidence']:.2%})")
    
    # Run streaming detection
    detector.stream_detection(
        simulator.generate_flows(),
        callback=alert_callback,
        max_flows=max_flows
    )


def main():
    """Main inference function"""
    args = parse_args()
    
    logger.info("="*60)
    logger.info("DNS SPOOFING DETECTION - INFERENCE")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    detector = RealTimeDNSDetector(model_path=args.model)
    
    # Run inference
    if args.stream:
        logger.info("Running in STREAMING mode")
        stream_inference(detector, args.input, args.max_flows)
    else:
        logger.info("Running in BATCH mode")
        batch_inference(detector, args.input, args.output, args.max_flows)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)
