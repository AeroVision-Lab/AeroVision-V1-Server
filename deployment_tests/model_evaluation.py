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


class AeroVisionModelEvaluator:
    def __init__(self, base_url: str = "http://localhost:8000", 
                 labels_file: str = "/home/wlx/Aerovision-V1/data/labels.csv",
                 test_data_dir: str = "/home/wlx/Aerovision-V1/data/labeled"):
        self.base_url = base_url
        self.labels_file = labels_file
        self.test_data_dir = test_data_dir
        self.labels_df = self._load_labels()
        self.results = {
            "aircraft": {},
            "airline": {},
            "timing": {}
        }

    def _load_labels(self) -> pd.DataFrame:
        """Load ground truth labels."""
        df = pd.read_csv(self.labels_file)
        # Map typeid to typename and airlineid to airlinename
        self.type_map = {row["typeid"]: row["typename"] for _, row in df.iterrows()}
        self.airline_map = {row["airlineid"]: row["airlinename"] for _, row in df.iterrows()}
        return df

    def _get_test_images(self) -> List[str]:
        """Get all test image paths that have labels."""
        available_images = set()
        data_dir = Path(self.test_data_dir)
        for ext in ["*.jpg", "*.jpeg"]:
            for img_path in data_dir.glob(ext):
                available_images.add(img_path.name)
        
        # Filter to only images with labels
        labeled_images = []
        for filename in available_images:
            if filename in self.labels_df["filename"].values:
                labeled_images.append(str(data_dir / filename))
        
        return labeled_images

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    def _get_ground_truth(self, image_path: str) -> Tuple[str, str]:
        """Get ground truth labels for an image."""
        filename = Path(image_path).name
        row = self.labels_df[self.labels_df["filename"] == filename]
        if len(row) == 0:
            return None, None
        
        row = row.iloc[0]
        aircraft_type = row["typename"]
        airline = row["airlinename"]
        return aircraft_type, airline

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
        """Calculate evaluation metrics."""
        # Top-1 accuracy
        top1_correct = 0
        top5_correct = 0
        total = len(predictions)
        
        # Confusion matrices
        confusions = defaultdict(int)
        predictions_list = []
        gt_list = []

        for pred, (gt_aircraft, gt_airline) in zip(predictions, ground_truths):
            if pred is None or "data" not in pred:
                continue
            
            data = pred["data"]
            
            # Aircraft metrics
            if "aircraft" in data:
                pred_aircraft = data["aircraft"]["typename"]
                predictions_list.append(pred_aircraft)
                gt_list.append(gt_aircraft)
                
                if pred_aircraft == gt_aircraft:
                    top1_correct += 1
                
                # Top-5 accuracy
                if "top_predictions" in data["aircraft"]:
                    top5_aircrafts = [p["typename"] for p in data["aircraft"]["top_predictions"][:5]]
                    if gt_aircraft in top5_aircrafts:
                        top5_correct += 1
                
                # Track confusion
                confusions[(gt_aircraft, pred_aircraft)] += 1

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
        avg_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / sum(m["support"] for m in class_metrics.values()) if class_metrics else 0
        avg_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / sum(m["support"] for m in class_metrics.values()) if class_metrics else 0
        avg_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / sum(m["support"] for m in class_metrics.values()) if class_metrics else 0

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

    async def evaluate_aircraft_model(self, test_images: List[str], sample_size: int = None) -> Dict[str, Any]:
        """Evaluate aircraft classification model."""
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
            for i, image_path in enumerate(test_images):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(test_images)}")

                image_base64 = self._encode_image(image_path)
                gt_aircraft, gt_airline = self._get_ground_truth(image_path)

                success, result = await self._predict_aircraft(session, image_base64, top_k=5)
                
                if success:
                    predictions.append(result)
                    ground_truths.append((gt_aircraft, gt_airline))
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
        """Run complete model evaluation."""
        test_images = self._get_test_images()
        print(f"Found {len(test_images)} labeled test images")

        if sample_size:
            test_images = test_images[:sample_size]
            print(f"Using sample of {len(test_images)} images")

        # Evaluate aircraft model
        aircraft_metrics = await self.evaluate_aircraft_model(test_images)

        summary = {
            "test_type": "model_evaluation",
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
            "test_data": {
                "total_images": len(test_images),
                "labels_file": self.labels_file
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
    args = parser.parse_args()

    base_url = args.url
    if args.cpu:
        base_url = "http://localhost:8001"
        print("Testing CPU version at http://localhost:8001")
    elif args.gpu:
        print("Testing GPU version at http://localhost:8000")
    else:
        print(f"Testing API at {base_url}")

    evaluator = AeroVisionModelEvaluator(base_url=base_url)
    results = await evaluator.evaluate_all(sample_size=args.sample_size)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    aircraft = results["results"]["aircraft"]
    print(f"Aircraft Top-1 Accuracy: {aircraft['top1_accuracy']:.4f}")
    print(f"Aircraft Top-5 Accuracy: {aircraft['top5_accuracy']:.4f}")
    print(f"Aircraft F1 Score: {aircraft['f1_score']:.4f}")
    print(f"Aircraft Precision: {aircraft['precision']:.4f}")
    print(f"Aircraft Recall: {aircraft['recall']:.4f}")
    if "timing" in aircraft:
        print(f"Average Latency: {aircraft['timing']['avg_latency_ms']:.2f}ms")
        print(f"Throughput: {aircraft['timing']['throughput_images_per_second']:.2f} images/sec")
    
    evaluator.save_results()


if __name__ == "__main__":
    asyncio.run(main())
