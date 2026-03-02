#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Enhanced test harness for the async inference generation workflow.

This test validates:
1. End-to-end execution of the async inference pipeline
2. Successful completion without exceptions
3. Detection of threading errors (analogous to TaskGroup exceptions)
4. Verification of output artifacts (action chunks, predicted timesteps)
5. Proper cleanup and resource management

Usage:
    # Run the test locally
    pytest test_generation_workflow.py -v

    # Run with coverage
    pytest test_generation_workflow.py -v --cov=lerobot.async_inference --cov-report=html

    # Run with detailed logging
    pytest test_generation_workflow.py -v -s --log-cli-level=DEBUG
"""

from __future__ import annotations

import sys
import threading
import time
import traceback
from concurrent import futures
from queue import Queue
from typing import Any

import pytest
import torch

# Skip entire module if grpc is not available
pytest.importorskip("grpc")


class ThreadExceptionTracker:
    """
    Tracks exceptions from threads to detect errors similar to TaskGroup exceptions.

    This class provides similar functionality to asyncio.TaskGroup exception handling,
    but for threading-based concurrent operations.
    """

    def __init__(self):
        self.exceptions: list[tuple[str, Exception, str]] = []
        self.lock = threading.Lock()
        self._original_hook = None

    def __enter__(self):
        """Install the exception hook to catch threading errors."""
        self._original_hook = threading.excepthook
        threading.excepthook = self._exception_handler
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore the original exception hook."""
        threading.excepthook = self._original_hook
        return False

    def _exception_handler(self, args):
        """Handle thread exceptions."""
        with self.lock:
            self.exceptions.append((
                args.thread.name if hasattr(args, 'thread') else "Unknown",
                args.exc_value if hasattr(args, 'exc_value') else Exception("Unknown error"),
                "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
                if hasattr(args, 'exc_traceback') else ""
            ))

    def check_exceptions(self):
        """
        Check if any exceptions occurred and raise an assertion error if so.

        Raises:
            AssertionError: If any thread exceptions were captured.
        """
        with self.lock:
            if self.exceptions:
                error_msgs = []
                for thread_name, exc, tb in self.exceptions:
                    error_msgs.append(f"Thread '{thread_name}' raised: {exc}\n{tb}")
                raise AssertionError(
                    f"Thread exceptions detected (similar to TaskGroup exceptions):\n" +
                    "\n".join(error_msgs)
                )


class WorkflowMetrics:
    """Collects metrics about the workflow execution."""

    def __init__(self):
        self.action_chunks_received = 0
        self.observations_sent = 0
        self.actions_executed = 0
        self.predicted_timesteps = set()
        self.errors: list[str] = []
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.lock = threading.Lock()

    def record_action_chunk(self):
        """Record that an action chunk was received."""
        with self.lock:
            self.action_chunks_received += 1

    def record_observation(self):
        """Record that an observation was sent."""
        with self.lock:
            self.observations_sent += 1

    def record_action_execution(self):
        """Record that an action was executed."""
        with self.lock:
            self.actions_executed += 1

    def record_predicted_timestep(self, timestep: int):
        """Record a predicted timestep."""
        with self.lock:
            self.predicted_timesteps.add(timestep)

    def record_error(self, error: str):
        """Record an error."""
        with self.lock:
            self.errors.append(error)

    def start(self):
        """Mark the start of the workflow."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Mark the end of the workflow."""
        self.end_time = time.perf_counter()

    def duration(self) -> float:
        """Get the workflow duration in seconds."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def summary(self) -> dict[str, Any]:
        """Get a summary of the metrics."""
        return {
            "action_chunks_received": self.action_chunks_received,
            "observations_sent": self.observations_sent,
            "actions_executed": self.actions_executed,
            "predicted_timesteps_count": len(self.predicted_timesteps),
            "errors_count": len(self.errors),
            "duration_seconds": self.duration(),
            "errors": self.errors,
        }


def test_async_inference_workflow_comprehensive(monkeypatch):
    """
    Comprehensive end-to-end test of the async inference generation workflow.

    This test:
    1. Sets up a complete PolicyServer and RobotClient infrastructure
    2. Runs the workflow for a configurable duration
    3. Tracks all threading exceptions (analogous to TaskGroup exceptions)
    4. Validates successful completion with output artifacts
    5. Ensures proper cleanup of all resources
    """
    # Import grpc-dependent modules inside the test function
    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
    from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.transport import (
        services_pb2,  # type: ignore
        services_pb2_grpc,  # type: ignore
    )
    from tests.mocks.mock_robot import MockRobotConfig

    # ------------------------------------------------------------------
    # Setup: Create mock policy and configuration
    # ------------------------------------------------------------------
    class MockPolicy:
        """Lightweight mock policy for testing."""

        class _Config:
            robot_type = "dummy_robot"

            @property
            def image_features(self):
                return {}

        def __init__(self):
            self.config = self._Config()

        def to(self, *args, **kwargs):
            return self

        def model(self, batch):
            batch_size = len(batch["robot_type"])
            return torch.zeros(batch_size, 20, 6)

    # Initialize metrics and exception tracking
    metrics = WorkflowMetrics()
    exception_tracker = ThreadExceptionTracker()

    with exception_tracker:
        try:
            # ------------------------------------------------------------------
            # 1. Create and configure PolicyServer
            # ------------------------------------------------------------------
            policy_server_config = PolicyServerConfig(host="localhost", port=9999)
            policy_server = PolicyServer(policy_server_config)
            policy_server.policy = MockPolicy()
            policy_server.actions_per_chunk = 20
            policy_server.device = "cpu"
            policy_server.preprocessor = lambda obs: obs
            policy_server.postprocessor = lambda tensor: tensor

            # Configure robot and features
            robot_config = MockRobotConfig()
            mock_robot = make_robot_from_config(robot_config)
            lerobot_features = map_robot_keys_to_lerobot_features(mock_robot)
            policy_server.lerobot_features = lerobot_features
            policy_server.policy_type = "act"

            # Monkeypatch methods for deterministic testing
            def _fake_get_action_chunk(_self, _obs, _type="test"):
                action_dim = 6
                batch_size = 1
                actions_per_chunk = policy_server.actions_per_chunk
                return torch.zeros(batch_size, actions_per_chunk, action_dim)

            def _fake_send_policy_instructions(self, request, context):
                return services_pb2.Empty()

            monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)
            monkeypatch.setattr(
                PolicyServer, "SendPolicyInstructions", _fake_send_policy_instructions, raising=True
            )

            # Start gRPC server
            server = grpc.server(
                futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="policy_server")
            )
            services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
            server_address = f"{policy_server.config.host}:{policy_server.config.port}"
            server.add_insecure_port(server_address)
            server.start()

            # ------------------------------------------------------------------
            # 2. Create and configure RobotClient
            # ------------------------------------------------------------------
            client_config = RobotClientConfig(
                server_address=server_address,
                robot=robot_config,
                chunk_size_threshold=0.0,
                policy_type="test",
                pretrained_name_or_path="test",
                actions_per_chunk=20,
            )

            client = RobotClient(client_config)
            assert client.start(), "Client failed initial handshake with the server"

            # Instrument client to track metrics
            original_aggregate = client._aggregate_action_queues

            def counting_aggregate(*args, **kwargs):
                metrics.record_action_chunk()
                return original_aggregate(*args, **kwargs)

            original_send_observation = client.send_observation

            def counting_send_observation(*args, **kwargs):
                result = original_send_observation(*args, **kwargs)
                if result:
                    metrics.record_observation()
                return result

            monkeypatch.setattr(client, "_aggregate_action_queues", counting_aggregate)
            monkeypatch.setattr(client, "send_observation", counting_send_observation)

            # ------------------------------------------------------------------
            # 3. Run the workflow for a configurable duration
            # ------------------------------------------------------------------
            workflow_duration = 5.0  # seconds
            metrics.start()

            # Start client threads
            action_thread = threading.Thread(
                target=client.receive_actions,
                daemon=True,
                name="action_receiver"
            )
            control_thread = threading.Thread(
                target=client.control_loop,
                args=({"task": "test_task"},),
                daemon=True,
                name="control_loop"
            )

            action_thread.start()
            control_thread.start()

            # Wait for workflow to run
            server.wait_for_termination(timeout=workflow_duration)
            metrics.stop()

            # ------------------------------------------------------------------
            # 4. Validate workflow execution and outputs
            # ------------------------------------------------------------------

            # Check that the workflow produced outputs
            assert metrics.action_chunks_received > 0, (
                f"No action chunks received. Workflow may have failed. "
                f"Metrics: {metrics.summary()}"
            )

            assert len(policy_server._predicted_timesteps) > 0, (
                f"Server did not record any predicted timesteps. "
                f"Server may not have processed observations. "
                f"Metrics: {metrics.summary()}"
            )

            # Validate that observations were sent
            assert metrics.observations_sent > 0, (
                f"No observations were sent to the server. "
                f"Client may have failed. "
                f"Metrics: {metrics.summary()}"
            )

            # Check for reasonable workflow progression
            assert metrics.action_chunks_received >= 1, (
                f"Insufficient action chunks received ({metrics.action_chunks_received}). "
                f"Expected at least 1 for a {workflow_duration}s workflow."
            )

            # Validate that predicted timesteps match expected behavior
            predicted_count = len(policy_server._predicted_timesteps)
            assert predicted_count >= 1, (
                f"Only {predicted_count} timesteps predicted. "
                f"Expected more activity in {workflow_duration}s."
            )

            # Log success metrics
            print(f"\n✅ Workflow completed successfully!")
            print(f"Metrics: {metrics.summary()}")
            print(f"Predicted timesteps: {sorted(policy_server._predicted_timesteps)}")

        except Exception as e:
            metrics.record_error(f"Workflow exception: {e}")
            metrics.stop()
            raise

        finally:
            # ------------------------------------------------------------------
            # 5. Cleanup resources
            # ------------------------------------------------------------------
            try:
                client.stop()
            except Exception as e:
                metrics.record_error(f"Client stop error: {e}")

            try:
                action_thread.join(timeout=2.0)
                control_thread.join(timeout=2.0)
            except Exception as e:
                metrics.record_error(f"Thread join error: {e}")

            try:
                policy_server.stop()
            except Exception as e:
                metrics.record_error(f"PolicyServer stop error: {e}")

            try:
                server.stop(grace=1.0)
            except Exception as e:
                metrics.record_error(f"gRPC server stop error: {e}")

    # ------------------------------------------------------------------
    # 6. Check for threading exceptions (analogous to TaskGroup)
    # ------------------------------------------------------------------
    exception_tracker.check_exceptions()

    # Final validation: no errors during execution or cleanup
    assert len(metrics.errors) == 0, (
        f"Errors occurred during workflow execution or cleanup: {metrics.errors}"
    )

    print(f"\n✅ All validations passed! No threading exceptions detected.")


def test_async_inference_workflow_stress(monkeypatch):
    """
    Stress test for the async inference workflow.

    This test runs a longer workflow to detect race conditions and
    resource leaks that might not appear in short tests.
    """
    import grpc

    from lerobot.async_inference.configs import PolicyServerConfig, RobotClientConfig
    from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
    from lerobot.async_inference.policy_server import PolicyServer
    from lerobot.async_inference.robot_client import RobotClient
    from lerobot.robots.utils import make_robot_from_config
    from lerobot.transport import (
        services_pb2,  # type: ignore
        services_pb2_grpc,  # type: ignore
    )
    from tests.mocks.mock_robot import MockRobotConfig

    class MockPolicy:
        class _Config:
            robot_type = "dummy_robot"

            @property
            def image_features(self):
                return {}

        def __init__(self):
            self.config = self._Config()

        def to(self, *args, **kwargs):
            return self

        def model(self, batch):
            batch_size = len(batch["robot_type"])
            return torch.zeros(batch_size, 20, 6)

    exception_tracker = ThreadExceptionTracker()

    with exception_tracker:
        policy_server_config = PolicyServerConfig(host="localhost", port=9998)
        policy_server = PolicyServer(policy_server_config)
        policy_server.policy = MockPolicy()
        policy_server.actions_per_chunk = 20
        policy_server.device = "cpu"
        policy_server.preprocessor = lambda obs: obs
        policy_server.postprocessor = lambda tensor: tensor

        robot_config = MockRobotConfig()
        mock_robot = make_robot_from_config(robot_config)
        lerobot_features = map_robot_keys_to_lerobot_features(mock_robot)
        policy_server.lerobot_features = lerobot_features
        policy_server.policy_type = "act"

        def _fake_get_action_chunk(_self, _obs, _type="test"):
            return torch.zeros(1, 20, 6)

        def _fake_send_policy_instructions(self, request, context):
            return services_pb2.Empty()

        monkeypatch.setattr(PolicyServer, "_get_action_chunk", _fake_get_action_chunk, raising=True)
        monkeypatch.setattr(
            PolicyServer, "SendPolicyInstructions", _fake_send_policy_instructions, raising=True
        )

        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
        server_address = f"{policy_server.config.host}:{policy_server.config.port}"
        server.add_insecure_port(server_address)
        server.start()

        client_config = RobotClientConfig(
            server_address=server_address,
            robot=robot_config,
            chunk_size_threshold=0.0,
            policy_type="test",
            pretrained_name_or_path="test",
            actions_per_chunk=20,
        )

        client = RobotClient(client_config)
        assert client.start()

        action_thread = threading.Thread(target=client.receive_actions, daemon=True)
        control_thread = threading.Thread(
            target=client.control_loop,
            args=({"task": "stress_test"},),
            daemon=True
        )

        action_thread.start()
        control_thread.start()

        # Run for longer duration to stress test
        server.wait_for_termination(timeout=10.0)

        # Validate continued operation
        assert len(policy_server._predicted_timesteps) > 5, (
            "Stress test should produce multiple predictions"
        )

        # Cleanup
        client.stop()
        action_thread.join(timeout=2.0)
        control_thread.join(timeout=2.0)
        policy_server.stop()
        server.stop(grace=1.0)

    exception_tracker.check_exceptions()
    print("✅ Stress test completed successfully!")


if __name__ == "__main__":
    """
    Allow running the test harness directly.

    Usage:
        python test_generation_workflow.py
    """
    pytest.main([__file__, "-v", "-s"])
