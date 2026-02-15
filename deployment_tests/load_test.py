#!/usr/bin/env python3
"""
High concurrency load testing script for AeroVision API.
Performs stepped load testing: 1 → 2 → 4 → 8 → 16 → 32 → 64 concurrent users.
"""

import asyncio
import aiohttp
import json
import time
import os
import base64
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import argparse


class AeroVisionLoadTester:
    def __init__(self, base_url: str = "http://localhost:8000", test_data_dir: str = None):
        """
        Initialize load tester.
        
        Args:
            base_url: API base URL
            test_data_dir: Directory containing test images (defaults to /home/wlx/Aerovision-V1/data/labeled)
        """
        self.base_url = base_url
        self.test_data_dir = test_data_dir or "/home/wlx/Aerovision-V1/data/labeled"
        self.test_images = self._get_test_images(max_images=100)
        self.results = []

    def _get_test_images(self, max_images: int = 100) -> List[str]:
        """Get test image paths."""
        data_dir = Path(self.test_data_dir)
        images = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.jpeg"))
        return [str(img) for img in images[:max_images]]

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    async def _single_request(self, session: aiohttp.ClientSession, endpoint: str, image_base64: str) -> Dict[str, Any]:
        """Execute a single API request."""
        start_time = time.time()
        result = {
            "success": False,
            "status_code": None,
            "latency_ms": 0,
            "error": None
        }

        try:
            payload = {
                "image": image_base64
            }
            async with session.post(f"{self.base_url}{endpoint}", json=payload) as response:
                result["status_code"] = response.status
                if response.status == 200:
                    await response.json()
                    result["success"] = True
                else:
                    result["error"] = f"HTTP {response.status}"
        except Exception as e:
            result["error"] = str(e)
        finally:
            result["latency_ms"] = (time.time() - start_time) * 1000
            return result

    async def _run_concurrent_test(self, session: aiohttp.ClientSession, concurrent_users: int, duration_seconds: int = 30) -> Dict[str, Any]:
        """Run test with specified concurrent users."""
        print(f"Testing with {concurrent_users} concurrent users...")

        start_time = time.time()
        end_time = start_time + duration_seconds
        all_results = []
        request_count = 0

        endpoints = [
            "/api/v1/aircraft",
            "/api/v1/airline",
            "/api/v1/quality"
        ]

        tasks = []

        async def continuous_requests():
            nonlocal request_count
            while time.time() < end_time:
                image_path = self.test_images[request_count % len(self.test_images)]
                image_base64 = self._encode_image(image_path)
                endpoint = endpoints[request_count % len(endpoints)]
                
                result = await self._single_request(session, endpoint, image_base64)
                all_results.append(result)
                request_count += 1
                
                # Small delay to prevent overwhelming
                await asyncio.sleep(0.01)

        # Start concurrent tasks
        for _ in range(concurrent_users):
            tasks.append(asyncio.create_task(continuous_requests()))

        # Wait for all tasks
        await asyncio.gather(*tasks)

        # Calculate statistics
        successful = [r for r in all_results if r["success"]]
        failed = [r for r in all_results if not r["success"]]
        latencies = [r["latency_ms"] for r in successful]

        stats = {
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_requests": len(all_results),
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "requests_per_second": len(all_results) / duration_seconds,
            "success_rate": len(successful) / len(all_results) if all_results else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
            "errors": {}
        }

        # Aggregate errors
        for result in failed:
            error = result["error"]
            stats["errors"][error] = stats["errors"].get(error, 0) + 1

        self.results.append(stats)
        print(f"  ✓ {stats['requests_per_second']:.2f} req/s | Success: {stats['success_rate']*100:.1f}% | Avg Latency: {stats['avg_latency_ms']:.1f}ms")

        return stats

    async def run_stepped_load_test(self, concurrency_levels: List[int] = [1, 2, 4, 8, 16, 32, 64], duration_per_level: int = 30):
        """Run stepped load test from low to high concurrency."""
        print("=" * 60)
        print("Starting Stepped Load Test")
        print("=" * 60)

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            # First, check if API is healthy
            try:
                async with session.get(f"{self.base_url}/api/v1/health") as response:
                    if response.status != 200:
                        print(f"API Health check failed: {response.status}")
                        return
                    print("✓ API is healthy")
            except Exception as e:
                print(f"✗ API Health check failed: {e}")
                return

            print()

            for level in concurrency_levels:
                await self._run_concurrent_test(session, level, duration_per_level)
                print()
                # Brief pause between tests
                await asyncio.sleep(5)

        print("=" * 60)
        print("Load Test Completed")
        print("=" * 60)

    def get_summary(self) -> Dict[str, Any]:
        """Get test summary."""
        return {
            "test_type": "stepped_load_test",
            "base_url": self.base_url,
            "timestamp": datetime.now().isoformat(),
            "concurrency_levels": [r["concurrent_users"] for r in self.results],
            "results": self.results,
            "summary": {
                "max_concurrent_users": max(r["concurrent_users"] for r in self.results),
                "max_rps": max(r["requests_per_second"] for r in self.results),
                "overall_success_rate": sum(r["successful_requests"] for r in self.results) / sum(r["total_requests"] for r in self.results) if self.results else 0,
                "avg_p95_latency": sum(r["p95_latency_ms"] for r in self.results) / len(self.results) if self.results else 0
            }
        }

    def save_results(self, output_file: str = "load_test_results.json"):
        """Save results to JSON file."""
        summary = self.get_summary()
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="AeroVision API Load Tester")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--cpu", action="store_true", help="Testing CPU version")
    parser.add_argument("--gpu", action="store_true", help="Testing GPU version")
    parser.add_argument("--duration", type=int, default=30, help="Duration per concurrency level in seconds")
    parser.add_argument("--data-dir", help="Test data directory with images (default: /home/wlx/Aerovision-V1/data/labeled)")
    args = parser.parse_args()

    base_url = args.url
    if args.cpu:
        base_url = "http://localhost:8001"
        print("Testing CPU version at http://localhost:8001")
    elif args.gpu:
        print("Testing GPU version at http://localhost:8000")
    else:
        print(f"Testing API at {base_url}")

    tester = AeroVisionLoadTester(base_url=base_url, test_data_dir=args.data_dir)
    await tester.run_stepped_load_test(duration_per_level=args.duration)
    tester.save_results(f"load_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")


if __name__ == "__main__":
    asyncio.run(main())
