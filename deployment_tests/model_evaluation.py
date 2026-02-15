#!/usr/bin/env python3
"""
Model evaluation script for AeroVision API.
Evaluates accuracy, precision, recall, F1, Top-1/Top-5 accuracy, and inference speed.
"""

import asyncio
import aiohttp
import json
import time
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path
import argparse
import pandas as pd
from collections import defaultdict

# Import ICAO code mapping
from icao_to_fullname_mapping import get_fullname


class AeroVisionModelEvaluator:
    def __init__(self, base_url: str = "http://localhost:8000",
                 test_data_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            base_url: API base URL
            test_data_dir: Test data directory containing images with ICAO codes in filenames
        """
        self.base_url = base_url
        self.test_data_dir = test_data_dir or "/home/wlx/Aerovision-V1/data/labeled"
        self.results = {
            "aircraft": {},
            "airline": {},
            "timing": {}
        }

    def _get_test_images(self) -> List[Tuple[str, str]]:
        """
        Get all test image paths with their ground truth labels.
        Labels are extracted from filenames using ICAO code format.
        
        Returns:
            List of tuples (image_path, icao_code)
        """
        images = []
        data_dir = Path(self.test_data_dir)
        
        # Support both .jpg and .jpeg files
        for ext in ["*.jpg", "*.jpeg"]:
            for img_path in data_dir.glob(ext):
                # Parse ICAO code from filename
                # Expected format: ICAOCODE-001.jpg or ICAOCODE_001.jpg
                filename = img_path.stem  # Remove extension
                icao_code = filename.split('-')[0].split('_')[0]
                
                if icao_code:  # Only add if we could extract a code
                    images.append((str(img_path), icao_code))
        
        return images

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _get_ground_truth(self, image_path: str, icao_code: str) -> Tuple[str, str]:
        """
        Get ground truth labels for an image.
        
        Args:
            image_path: Path to the image file
            icao_code: ICAO code extracted from filename
            
        Returns:
            Tuple of (expected_fullname, icao_code)
        """
        # Convert ICAO code to full name to match model output
        expected_fullname = get_fullname(icao_code)
        return expected_fullname, icao_code

    async def _predict_aircraft(self, session: aiohttp.ClientSession, image_base64: str, top_k: int = 5) -> Tuple[bool, Dict]:
        """Predict aircraft type."""
        start_time = time.time()
        try:
            payload = {"image": image_base64}
            params = {"top_k": top_k}
            async with session.post(f"{self.base_url}/api/v1/aircraft", json=payload, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = (time.time() - start_time) * 1000
                    return True, {"data": data, "latency_ms": latency}
                else:
                    return False, {"error": f"HTTP {response.status}", "latency_ms": (time.time() - start_time) * 1000}
        except Exception as e:
            return False, {"error": str(e), "latency_ms": (time.time() - start_time) * 1000}

    async def _predict_airline(self, session: aiohttp.ClientSession, image_base64: str, top_k: int = 5) -> Tuple[bool, Dict]:
        """Predict airline."""
        start_time = time.time()
        try:
            payload = {"image": image_base64}
            params = {"top_k": top_k}
            async with session.post(f"{self.base_url}/api/v1/airline", json=payload, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    latency = (time.time() - start_time) * 1000
                    return True, {"data": data, "latency_ms": latency}
                else:
                    return False, {"error": f"HTTP {response.status}", "latency_ms": (time.time() - start_time) * 1000}
        except Exception as e:
            return False, {"error": str(e), "latency_ms": (time.time() - start_time) * 1000}

    def _calculate_metrics(self, predictions: List[Dict], ground_truths: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.
        
        Args:
            predictions: List of prediction results from API
            ground_truths: List of tuples (expected_fullname, icao_code)
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Top-1 accuracy
        top1_correct = 0
        top5_correct = 0
        total = len(predictions)
        
        # Confusion matrices
        confusions = defaultdict(int)
        predictions_list = []
        gt_list = []
        icao_gt_list = []  # Track original ICAO codes

        for pred, (gt_fullname, icao_code) in zip(predictions, ground_truths):
            if pred is None or "data" not in pred:
                continue
            
            data = pred["data"]
            
            # Aircraft metrics
            pred_aircraft = data.get("top1", {}).get("class", "")
            predictions_list.append(pred_aircraft)
            gt_list.append(gt_fullname)
            icao_gt_list.append(icao_code)
            
            if pred_aircraft == gt_fullname:
                top1_correct += 1
            
            # Top-5 accuracy
            predictions = data.get("predictions", [])
            top5_aircrafts = [p["class"] for p in predictions[:5]]
            if gt_fullname in top5_aircrafts:
                top5_correct += 1
            
            # Track confusion
            confusions[(gt_fullname, pred_aircraft)] += 1

        # Calculate per-class metrics
        class_metrics = {}
        all_classes = set(gt_list + predictions_list)
        
        for cls in all_classes:
            true_positives = sum(1 for gt, pred in zip(gt_list, predictions_list) if gt == cls and pred == cls)
            false_positives = sum(1 for gt, pred in zip(gt_list, predictions_list) if gt != cls and pred == cls)
            false_negatives = sum(1 for gt, pred in zip(gt_list, predictions_list) if gt == cls and pred != cls)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": sum(1 for gt in gt_list if gt == cls)
            }

        # Calculate average metrics
        total_support = sum(m["support"] for m in class_metrics.values())
        avg_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
        avg_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0
        avg_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0

        return {
            "total_samples": total,
            "top1_accuracy": top1_correct / total if total > 0 else 0,
            "top5_accuracy": top5_correct / total if total > 0 else 0,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "per_class_metrics": class_metrics,
            "confusion_matrix_sample": dict(list(confusions.items())[:10])  # Sample of confusions
        }

    async def evaluate_aircraft_model(self, test_images: List[Tuple[str, str]], sample_size: int = None) -> Dict[str, Any]:
        """
        Evaluate aircraft classification model.
        
        Args:
            test_images: List of tuples (image_path, icao_code)
            sample_size: Optional sample size to limit testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 60)
        print("Evaluating Aircraft Classification Model")
        print("=" * 60)

        if sample_size:
            test_images = test_images[:sample_size]

        print(f"Testing {len(test_images)} images...")

        predictions = []
        ground_truths = []
        latencies = []
        errors = []

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            for i, (image_path, icao_code) in enumerate(test_images):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(test_images)}")

                image_base64 = self._encode_image(image_path)
                gt_fullname, gt_icao = self._get_ground_truth(image_path, icao_code)

                success, result = await self._predict_aircraft(session, image_base64, top_k=5)
                
                if success:
                    predictions.append(result)
                    ground_truths.append((gt_fullname, gt_icao))
                    latencies.append(result["latency_ms"])
                else:
                    errors.append(result["error"])

        print(f"  ✓ Completed: {len(predictions)}/{len(test_images)} successful")
        print(f"  ✗ Errors: {len(errors)}")

        metrics = self._calculate_metrics(predictions, ground_truths)
        
        # Timing statistics
        if latencies:
            metrics["timing"] = {
                "avg_latency_ms": sum(latencies) / len(latencies),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2],
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
                "throughput_images_per_second": len(predictions) / (sum(latencies) / 1000) if latencies else 0
            }

        metrics["errors"] = errors[:10]  # Sample of errors
        self.results["aircraft"] = metrics

        return metrics

    async def evaluate_all(self, sample_size: int = 200) -> Dict[str, Any]:
        """
        Run complete model evaluation.
        
        Args:
            sample_size: Number of images to test (None for all)
            
        Returns:
            Dictionary containing all evaluation results
        """
        test_images = self._get_test_images()
        print(f"Found {len(test_images)} test images")

        if sample_size:
            test_images = test_images[:sample_size]
            print(f"Using sample of {len(test_images)} images")

        if not test_images:
            print("⚠️  No test images found!")
            return {
                "test_type": "model_evaluation",
                "base_url": self.base_url,
                "timestamp": datetime.now().isoformat(),
                "error": "No test images found in " + self.test_data_dir,
                "results": self.results
            }

        # Evaluate aircraft model
        aircraft_metrics = await self.evaluate_aircraft_model(test_images)

        summary = {
            "test_type": "model_evaluation",
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
            "test_data": {
                "total_images": len(test_images),
                "test_data_dir": self.test_data_dir
            },
            "results": self.results,
            "summary": {
                "aircraft_top1_accuracy": aircraft_metrics.get("top1_accuracy", 0),
                "aircraft_top5_accuracy": aircraft_metrics.get("top5_accuracy", 0),
                "aircraft_f1": aircraft_metrics.get("f1_score", 0),
                "avg_latency_ms": aircraft_metrics.get("timing", {}).get("avg_latency_ms", 0)
            }
        }

        return summary

    def save_results(self, output_file: str = None):
        """Save results to JSON file."""
        if output_file is None:
            output_file = f"model_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            "test_type": "model_evaluation",
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
            "results": self.results
        }
        
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n✓ Results saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="AeroVision Model Evaluator")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--cpu", action="store_true", help="Testing CPU version")
    parser.add_argument("--gpu", action="store_true", help="Testing GPU version")
    parser.add_argument("--sample-size", type=int, default=200, help="Number of images to test")
    parser.add_argument("--data-dir", help="Test data directory with ICAO-coded images (default: /home/wlx/Aerovision-V1/data/labeled)")
    args = parser.parse_args()

    base_url = args.url
    if args.cpu:
        base_url = "http://localhost:8001"
        print("Testing CPU version at http://localhost:8001")
    elif args.gpu:
        base_url = "http://localhost:8002"
        print("Testing GPU version at http://localhost:8002")
    else:
        print(f"Testing API at {base_url}")

    evaluator = AeroVisionModelEvaluator(base_url=base_url, test_data_dir=args.data_dir)
    results = await evaluator.evaluate_all(sample_size=args.sample_size)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    if "error" in results:
        print(f"❌ Error: {results['error']}")
        return
    
    aircraft = results["results"]["aircraft"]
    print(f"Aircraft Top-1 Accuracy: {aircraft['top1_accuracy']:.4f} ({aircraft['top1_accuracy']*100:.2f}%)")
    print(f"Aircraft Top-5 Accuracy: {aircraft['top5_accuracy']:.4f} ({aircraft['top5_accuracy']*100:.2f}%)")
    print(f"Aircraft F1 Score: {aircraft['f1_score']:.4f}")
    print(f"Aircraft Precision: {aircraft['precision']:.4f}")
    print(f"Aircraft Recall: {aircraft['recall']:.4f}")
    if "timing" in aircraft:
        print(f"Average Latency: {aircraft['timing']['avg_latency_ms']:.2f}ms")
        print(f"Throughput: {aircraft['timing']['throughput_images_per_second']:.2f} images/sec")
    
    evaluator.save_results()


if __name__ == "__main__":
    asyncio.run(main())
