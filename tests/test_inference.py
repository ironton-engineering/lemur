import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from server import analyzer, codex, config, main, processes
from server.gpu import GPU


def _server(**overrides):
    values = {
        "id": "test",
        "model_path": "/tmp/model.gguf",
        "model_name": "model.gguf",
        "gpu": 0,
        "port": 8080,
        "ctx": 8192,
        "ngl": 999,
        "host": "127.0.0.1",
        "spill": "none",
        "mtp": True,
        "mtp_draft_n": 2,
    }
    values.update(overrides)
    return processes.ServerInstance(**values)


class CapacityTests(unittest.TestCase):
    def test_start_server_allows_vram_capacity_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            binary = Path(tmp) / "llama-server"
            model.write_bytes(b"GGUF")
            binary.write_text("#!/bin/sh\n")
            info = SimpleNamespace(
                format="gguf", name=model.name, alias=None, has_mmproj=False
            )
            settings = {
                "llama_server_path": str(binary),
                "default_ctx": 8192,
                "default_ngl": 999,
                "default_host": "127.0.0.1",
                "port_start": 8080,
            }
            thread = MagicMock()
            with (
                patch.object(processes, "get_settings", return_value=settings),
                patch.object(processes.model_scan, "find_model", return_value=info),
                patch.object(processes.gguf_meta, "read_gguf_arch", return_value={}),
                patch.object(processes, "_device_list", return_value=[0]),
                patch.object(processes, "_port_free", return_value=True),
                patch.object(
                    processes,
                    "_validate_cuda_capacity",
                    side_effect=processes.CapacityWarning("estimated OOM"),
                ),
                patch.object(processes.threading, "Thread", return_value=thread),
            ):
                server = processes.start_server(
                    str(model), gpu=0, port=18080, spill="none"
                )

        self.assertEqual(server.status, "starting")
        self.assertIn("launch override is enabled", "\n".join(server.logs))
        thread.start.assert_called_once()

    def test_borderline_full_offload_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            with (
                patch.object(
                    processes.gguf_meta,
                    "read_gguf_arch",
                    return_value={"n_layer": 64},
                ),
                patch.object(
                    processes.gguf_meta,
                    "estimate_vram_mib",
                    return_value={"total_mib": 10750.0},
                ),
                patch.object(
                    processes.gpu_mod,
                    "list_gpus",
                    return_value=[GPU(0, "small", 12288, memory_free_mib=11000)],
                ),
            ):
                processes._validate_cuda_capacity(_server(mtp=False), model)

    def test_impossible_full_offload_is_rejected_before_cuda(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            with (
                patch.object(
                    processes.gguf_meta,
                    "read_gguf_arch",
                    return_value={"n_layer": 64},
                ),
                patch.object(
                    processes.gguf_meta,
                    "estimate_vram_mib",
                    return_value={"total_mib": 26000.0},
                ),
                patch.object(
                    processes.gpu_mod,
                    "list_gpus",
                    return_value=[GPU(0, "small", 12288, memory_free_mib=11000)],
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "full offload needs"):
                    processes._validate_cuda_capacity(_server(mtp=False), model)

    def test_ram_fallback_bypasses_gpu_only_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            with patch.object(processes.gpu_mod, "list_gpus") as list_gpus:
                processes._validate_cuda_capacity(_server(spill="ram"), model)
                list_gpus.assert_not_called()

    def test_mtp_rejects_a_gguf_without_prediction_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            with patch.object(
                processes.gguf_meta,
                "read_gguf_arch",
                return_value={"n_layer": 64, "mtp_capable": False},
            ):
                with self.assertRaisesRegex(processes.PlacementError, "no next-token"):
                    processes._validate_cuda_capacity(_server(spill="ram"), model)

    def test_unsafe_multi_gpu_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            cards = [
                GPU(0, "large", 32000, memory_free_mib=30000),
                GPU(1, "small", 12000, memory_free_mib=5000),
            ]
            with (
                patch.object(
                    processes.gguf_meta,
                    "read_gguf_arch",
                    return_value={"n_layer": 64},
                ),
                patch.object(
                    processes.gguf_meta,
                    "estimate_vram_mib",
                    return_value={"total_mib": 26000.0},
                ),
                patch.object(processes.gpu_mod, "list_gpus", return_value=cards),
            ):
                with self.assertRaisesRegex(RuntimeError, "multi-GPU split"):
                    processes._validate_cuda_capacity(
                        _server(spill="gpu", mtp=False), model
                    )

    def test_estimator_treats_ram_as_a_slow_fallback_not_an_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            estimate = {
                "weights_mib": 24000.0,
                "kv_mib": 500.0,
                "scratch_mib": 1000.0,
                "mtp_mib": 500.0,
                "total_mib": 26000.0,
                "confidence": "high",
            }
            with (
                patch.object(main.models, "find_model", return_value=None),
                patch.object(main.gguf_meta, "read_gguf_arch", return_value={}),
                patch.object(main.gguf_meta, "estimate_vram_mib", return_value=estimate),
                patch.object(
                    main.gpu,
                    "list_gpus",
                    return_value=[GPU(1, "small", 12288, memory_free_mib=11000)],
                ),
                patch.object(main.gpu, "available_ram_mib", return_value=50000.0),
            ):
                result = main.api_vram_estimate(
                    main.VramEstimateRequest(
                        model=str(model), gpu=1, spill="ram", mtp=True
                    )
                )

        self.assertTrue(result["fits"])
        self.assertFalse(result["fits_vram"])
        self.assertTrue(result["uses_ram"])
        self.assertEqual(result["verdict"], "tight")
        self.assertIn("spill to system RAM", result["tip"])


class CommandTests(unittest.TestCase):
    def test_qwen38_nvfp4_discovery_sets_mtp_and_vision_defaults(self):
        from server import models as model_scan

        name = "Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / name
            projector = Path(tmp) / "mmproj-BF16.gguf"
            model.write_bytes(b"GGUF")
            projector.write_bytes(b"projector")
            found = model_scan._group_ggufs([(name, model, model.stat().st_size)])

        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].has_mmproj)
        self.assertEqual(found[0].mmproj_name, "mmproj-BF16.gguf")
        self.assertEqual(found[0].recommended_mtp_draft_n, 3)

    def test_qwen35_moe_uses_hybrid_memory_and_runtime_profile(self):
        from server import gguf_meta

        self.assertTrue(gguf_meta.is_qwen35_arch({"architecture": "qwen35moe"}))
        self.assertEqual(gguf_meta._attention_layer_count("qwen35moe", 40), 10)
        est = gguf_meta.estimate_vram_mib(
            weights_bytes=24 * 1024**3,
            ctx=131072,
            arch={
                "n_layer": 41,
                "n_attn_layer": 10,
                "n_head_kv": 2,
                "head_dim": 256,
            },
            mtp=True,
            kv_bytes_per_elem=1,
        )
        self.assertLess(est["total_mib"] / 1024.0, 31.0)

    def test_mtp_forces_supported_single_slot_flash_attention(self):
        proc = MagicMock(pid=123)
        thread = MagicMock()
        env = {}
        with (
            patch.object(processes, "_validate_cuda_capacity"),
            patch.object(processes, "_device_list", return_value=[0]),
            patch.object(
                processes.gguf_meta,
                "read_gguf_arch",
                return_value={"architecture": "qwen35"},
            ),
            patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
            patch.object(processes.threading, "Thread", return_value=thread),
        ):
            processes._spawn_llama(
                _server(), Path("/tmp/model.gguf"), "/tmp/llama-server", env
            )

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-fa") + 1], "on")
        self.assertEqual(cmd[cmd.index("-np") + 1], "1")
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "2")
        self.assertEqual(cmd[cmd.index("--temp") + 1], "0.6")
        self.assertEqual(cmd[cmd.index("--top-k") + 1], "20")
        self.assertEqual(cmd[cmd.index("--min-p") + 1], "0.0")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertNotIn("GGML_CUDA_DISABLE_GRAPHS", env)
        self.assertNotIn("GGML_CUDA_PDL", env)

    def test_qwen38_nvfp4_uses_publisher_speculative_profile(self):
        proc = MagicMock(pid=124)
        thread = MagicMock()
        env = {}
        name = "Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / name
            model.write_bytes(b"GGUF")
            arch = {
                "architecture": "qwen35",
                "n_layer": 65,
                "nextn_predict_layers": 1,
                "mtp_capable": True,
            }
            with (
                patch.object(processes, "_validate_cuda_capacity"),
                patch.object(processes, "_device_list", return_value=[0]),
                patch.object(processes.gguf_meta, "read_gguf_arch", return_value=arch),
                patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
                patch.object(processes.threading, "Thread", return_value=thread),
            ):
                processes._spawn_llama(
                    _server(
                        model_path=str(model),
                        model_name=name,
                        mtp=True,
                        mtp_draft_n=3,
                    ),
                    model,
                    "/tmp/llama-server",
                    env,
                )

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(cmd[cmd.index("--spec-draft-p-split") + 1], "0.2")
        self.assertEqual(cmd[cmd.index("--temp") + 1], "1.0")
        self.assertEqual(cmd[cmd.index("--top-p") + 1], "0.95")
        self.assertEqual(cmd[cmd.index("--top-k") + 1], "20")
        self.assertNotIn("--min-p", cmd)

    def test_qwen38_nvfp4_uses_q8_kv_at_long_context(self):
        proc = MagicMock(pid=125)
        thread = MagicMock()
        name = "Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
        model_size = 19_694_360_544
        arch = {
            "architecture": "qwen35",
            "n_layer": 65,
            "nextn_predict_layers": 1,
            "mtp_capable": True,
        }
        with (
            patch.object(processes, "_validate_cuda_capacity"),
            patch.object(processes, "_device_list", return_value=[0]),
            patch.object(
                processes.gpu_mod,
                "list_gpus",
                return_value=[GPU(0, "RTX 5090", 32607, memory_free_mib=32000)],
            ),
            patch.object(processes.gguf_meta, "read_gguf_arch", return_value=arch),
            patch.object(
                processes.gguf_meta,
                "estimate_vram_mib",
                return_value={"total_mib": 30000.0, "kv_mib": 5000.0},
            ),
            patch.object(
                Path,
                "stat",
                return_value=SimpleNamespace(st_size=model_size),
            ),
            patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
            patch.object(processes.threading, "Thread", return_value=thread),
        ):
            processes._spawn_llama(
                _server(
                    model_path=f"/tmp/{name}",
                    model_name=name,
                    ctx=131072,
                    spill="ram",
                    mtp=True,
                    mtp_draft_n=3,
                ),
                Path(f"/tmp/{name}"),
                "/tmp/llama-server",
                {},
            )

        cmd = popen.call_args.args[0]
        for flag in ("-ctk", "-ctv", "-ctkd", "-ctvd"):
            self.assertEqual(cmd[cmd.index(flag) + 1], "q8_0")
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "999")
        self.assertEqual(cmd[cmd.index("--fit") + 1], "off")

    def test_vision_attaches_sibling_mmproj(self):
        proc = MagicMock(pid=123)
        thread = MagicMock()
        env = {}
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            mmproj = Path(tmp) / "mmproj-F16.gguf"
            model.write_bytes(b"GGUF")
            mmproj.write_bytes(b"MMPROJ")
            with (
                patch.object(processes, "_validate_cuda_capacity"),
                patch.object(processes, "_device_list", return_value=[0]),
                patch.object(
                    processes.gguf_meta,
                    "read_gguf_arch",
                    return_value={"architecture": "qwen35"},
                ),
                patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
                patch.object(processes.threading, "Thread", return_value=thread),
            ):
                processes._spawn_llama(
                    _server(mtp=False, vision=True),
                    model,
                    "/tmp/llama-server",
                    env,
                )

            cmd = popen.call_args.args[0]
            self.assertEqual(cmd[cmd.index("--mmproj") + 1], str(mmproj))
            self.assertIn("--jinja", cmd)

    def test_vision_without_mmproj_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            model.write_bytes(b"GGUF")
            with (
                patch.object(processes, "_validate_cuda_capacity"),
                patch.object(processes, "_device_list", return_value=[0]),
                patch.object(
                    processes.gguf_meta,
                    "read_gguf_arch",
                    return_value={"architecture": "llama"},
                ),
            ):
                with self.assertRaises(FileNotFoundError):
                    processes._spawn_llama(
                        _server(mtp=False, vision=True),
                        model,
                        "/tmp/llama-server",
                        {},
                    )

    def test_qwen_long_context_uses_q8_kv_and_full_gpu_on_single_card(self):
        proc = MagicMock(pid=123)
        thread = MagicMock()
        env = {}
        with (
            patch.object(processes, "_validate_cuda_capacity"),
            patch.object(processes, "_device_list", return_value=[0]),
            patch.object(
                processes.gpu_mod,
                "list_gpus",
                return_value=[GPU(0, "RTX 5090", 32607, memory_free_mib=32000)],
            ),
            patch.object(
                processes.gguf_meta,
                "read_gguf_arch",
                return_value={"architecture": "qwen35"},
            ),
            patch.object(
                processes.gguf_meta,
                "estimate_vram_mib",
                return_value={"total_mib": 30000.0, "kv_mib": 5000.0},
            ),
            patch.object(
                Path,
                "stat",
                return_value=SimpleNamespace(st_size=24 * 1024**3),
            ),
            patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
            patch.object(processes.threading, "Thread", return_value=thread),
        ):
            processes._spawn_llama(
                _server(ctx=131072, spill="ram"),
                Path("/tmp/model.gguf"),
                "/tmp/llama-server",
                env,
            )

        cmd = popen.call_args.args[0]
        # The estimate fits with a reserve, so avoid CPU layer placement.
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "999")
        self.assertEqual(cmd[cmd.index("--fit") + 1], "off")
        self.assertNotIn("--fit-target", cmd)
        self.assertEqual(cmd[cmd.index("-ub") + 1], "256")
        for flag in ("-ctk", "-ctv", "-ctkd", "-ctvd"):
            self.assertEqual(cmd[cmd.index(flag) + 1], "q8_0")
        self.assertEqual(cmd.count("-fa"), 1)
        self.assertEqual(env["GGML_CUDA_DISABLE_GRAPHS"], "1")
        self.assertEqual(env["GGML_CUDA_PDL"], "0")

    def test_qwen_256k_keeps_ram_fit_on_single_card(self):
        proc = MagicMock(pid=123)
        thread = MagicMock()
        with (
            patch.object(processes, "_validate_cuda_capacity"),
            patch.object(processes, "_device_list", return_value=[0]),
            patch.object(
                processes.gpu_mod,
                "list_gpus",
                return_value=[GPU(0, "RTX 5090", 32607, memory_free_mib=32000)],
            ),
            patch.object(
                processes.gguf_meta,
                "read_gguf_arch",
                return_value={"architecture": "qwen35"},
            ),
            patch.object(
                Path,
                "stat",
                return_value=SimpleNamespace(st_size=24 * 1024**3),
            ),
            patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
            patch.object(processes.threading, "Thread", return_value=thread),
        ):
            processes._spawn_llama(
                _server(ctx=262144, spill="ram"),
                Path("/tmp/model.gguf"),
                "/tmp/llama-server",
                {},
            )

        cmd = popen.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "auto")
        self.assertEqual(cmd[cmd.index("--fit") + 1], "on")
        self.assertEqual(cmd[cmd.index("--fit-target") + 1], "2048")
        for flag in ("-ctk", "-ctv", "-ctkd", "-ctvd"):
            self.assertEqual(cmd[cmd.index(flag) + 1], "q8_0")

    def test_q8_profile_does_not_reduce_small_qwen_cache_quality(self):
        proc = MagicMock(pid=123)
        thread = MagicMock()
        with (
            patch.object(processes, "_validate_cuda_capacity"),
            patch.object(processes, "_device_list", return_value=[0]),
            patch.object(
                processes.gpu_mod,
                "list_gpus",
                return_value=[GPU(0, "RTX 5090", 32607, memory_free_mib=32000)],
            ),
            patch.object(
                processes.gguf_meta,
                "read_gguf_arch",
                return_value={"architecture": "qwen35"},
            ),
            patch.object(
                Path,
                "stat",
                return_value=SimpleNamespace(st_size=8 * 1024**3),
            ),
            patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
            patch.object(processes.threading, "Thread", return_value=thread),
        ):
            processes._spawn_llama(
                _server(ctx=131072, spill="ram"),
                Path("/tmp/model.gguf"),
                "/tmp/llama-server",
                {},
            )

        cmd = popen.call_args.args[0]
        self.assertNotIn("-ctk", cmd)
        self.assertNotIn("-ctv", cmd)

    def test_fit_target_favors_large_primary_over_small_secondary(self):
        cards = [
            GPU(0, "RTX 5090", 32607),
            GPU(1, "RTX 3060", 12288),
        ]
        with patch.object(processes.gpu_mod, "list_gpus", return_value=cards):
            self.assertEqual(processes._fit_target_for([0, 1]), "1024,3072")


class StreamTests(unittest.TestCase):
    def test_context_overflow_retry_halves_large_output_reserve(self):
        body = {"max_output_tokens": 32768, "stream": True}
        error = {
            "message": (
                "This model's maximum context length is 131072 tokens. "
                "However, you requested 32768 output tokens and your prompt "
                "contains at least 98305 input tokens."
            ),
            "type": "BadRequestError",
        }

        retry = main._context_overflow_retry(body, json.dumps(error).encode(), 400)

        self.assertIsNotNone(retry)
        self.assertEqual(retry["max_output_tokens"], 16384)
        self.assertEqual(body["max_output_tokens"], 32768)

    def test_context_overflow_retry_ignores_other_bad_requests(self):
        body = {"max_tokens": 32768}

        retry = main._context_overflow_retry(
            body, b'{"error":{"message":"invalid tool"}}', 400
        )

        self.assertIsNone(retry)

    def test_partial_sse_json_is_buffered_until_the_line_is_complete(self):
        with patch.object(main, "_ingest_stream_chunk") as ingest:
            pending = main._ingest_complete_sse_lines(
                "server", "trace", "/v1/chat/completions", b"", b'data: {"x"'
            )
            self.assertEqual(pending, b'data: {"x"')
            ingest.assert_not_called()

            pending = main._ingest_complete_sse_lines(
                "server", "trace", "/v1/chat/completions", pending, b": 1}\n\n"
            )

        self.assertEqual(pending, b"")
        ingest.assert_called_once_with(
            "server", "trace", "/v1/chat/completions", b'data: {"x": 1}\n\n'
        )


class CodexProfileTests(unittest.TestCase):
    def test_compaction_leaves_space_for_32k_output_at_128k_context(self):
        compact = codex._compact_limit(131072)

        self.assertEqual(compact, 97280)
        self.assertLess(compact + codex.DEFAULT_MAX_OUTPUT_TOKENS, 131072)


class NetworkAccessTests(unittest.TestCase):
    def test_enabling_lan_restarts_running_models_on_all_interfaces(self):
        running = _server(
            model_path="/models/a.gguf",
            gpu=1,
            port=8087,
            ctx=16384,
            ngl=42,
            spill="ram",
            mtp=False,
            vision=True,
            status="running",
        )
        replacement = _server(host="0.0.0.0", port=8087, status="starting")
        with (
            patch.object(processes, "list_servers", return_value=[running]),
            patch.object(processes, "stop_all_servers") as stop_all,
            patch.object(processes, "start_server", return_value=replacement) as start,
            patch.object(main.config, "update_settings") as update_settings,
            patch.object(main, "_sync_codex"),
        ):
            result = main.api_update_network_access(
                main.NetworkAccessUpdate(lan_enabled=True)
            )

        stop_all.assert_called_once_with()
        update_settings.assert_called_once_with({"default_host": "0.0.0.0"})
        self.assertEqual(start.call_args.kwargs["host"], "0.0.0.0")
        self.assertEqual(start.call_args.kwargs["port"], 8087)
        self.assertEqual(start.call_args.kwargs["ctx"], 16384)
        self.assertEqual(start.call_args.kwargs["spill"], "ram")
        self.assertTrue(start.call_args.kwargs["vision"])
        self.assertTrue(result["lan_enabled"])

    def test_network_change_waits_until_models_finish_starting(self):
        with patch.object(
            processes, "list_servers", return_value=[_server(status="starting")]
        ):
            with self.assertRaisesRegex(
                main.HTTPException, "finish starting"
            ) as raised:
                main.api_update_network_access(
                    main.NetworkAccessUpdate(lan_enabled=True)
                )

        self.assertEqual(raised.exception.status_code, 409)


class FavoriteTests(unittest.TestCase):
    def test_running_configuration_becomes_one_click_favorite(self):
        running = _server(
            model_path="/models/qwen.gguf",
            model_name="qwen.gguf",
            alias="qwen",
            format="gguf",
            gpu=1,
            ctx=131072,
            ngl=64,
            spill="ram",
            mtp=True,
            mtp_draft_n=3,
            vision=True,
            status="running",
        )
        replacement = _server(
            model_path=running.model_path,
            model_name=running.model_name,
            alias=running.alias,
            gpu=running.gpu,
            ctx=running.ctx,
            ngl=running.ngl,
            spill=running.spill,
            mtp=running.mtp,
            mtp_draft_n=running.mtp_draft_n,
            vision=running.vision,
            status="starting",
        )
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            with (
                patch.object(config, "STATE_FILE", state_file),
                patch.object(processes, "get_server", return_value=running),
                patch.object(
                    processes, "start_server", return_value=replacement
                ) as start,
                patch.object(main, "_sync_codex", return_value={"ok": True}),
            ):
                created = main.api_create_favorite(
                    main.FavoriteCreate(server_id=running.id)
                )
                duplicate = main.api_create_favorite(
                    main.FavoriteCreate(server_id=running.id)
                )
                favorite_id = created["favorite"]["id"]
                started = main.api_start_favorite(favorite_id)
                listed = main.api_favorites()
                deleted = main.api_delete_favorite(favorite_id)

        self.assertTrue(created["created"])
        self.assertFalse(duplicate["created"])
        self.assertEqual(listed["count"], 1)
        self.assertEqual(started["model_path"], running.model_path)
        self.assertEqual(start.call_args.kwargs["gpu"], 1)
        self.assertEqual(start.call_args.kwargs["ctx"], 131072)
        self.assertEqual(start.call_args.kwargs["spill"], "ram")
        self.assertTrue(start.call_args.kwargs["mtp"])
        self.assertEqual(start.call_args.kwargs["mtp_draft_n"], 3)
        self.assertTrue(start.call_args.kwargs["vision"])
        self.assertEqual(start.call_args.kwargs["format_hint"], "gguf")
        self.assertTrue(deleted["ok"])


class NativeVllmTests(unittest.TestCase):
    def test_detects_compressed_nvfp4_vlm(self):
        from server import models as model_scan

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Qwen3.8-27B-NVFP4"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                        "model_type": "qwen3_5",
                        "vision_config": {"depth": 27},
                        "text_config": {"mtp_num_hidden_layers": 1},
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "format": "mixed-precision",
                            "config_groups": {
                                "fp8": {"format": "float-quantized"},
                                "fp4": {"format": "nvfp4-pack-quantized"},
                            },
                        },
                    }
                )
            )
            (model / "model.safetensors").write_bytes(b"weights")

            self.assertTrue(model_scan._is_text_hf_dir(model))
            self.assertEqual(model_scan._hf_format(model), "vllm")
            self.assertEqual(model_scan._hf_native_features(model), (True, True))

    def test_spawn_vllm_uses_native_blackwell_features(self):
        proc = MagicMock(pid=987)
        thread = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Qwen3.8-27B-NVFP4"
            model.mkdir()
            binary = Path(tmp) / "vllm"
            binary.write_text("#!/bin/sh\n")
            server = _server(
                model_path=str(model),
                model_name=model.name,
                alias="Qwen3.8-27B-NVFP4",
                format="vllm",
                vision=True,
                mtp=True,
                mtp_draft_n=2,
            )
            with (
                patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
                patch.object(processes.threading, "Thread", return_value=thread),
            ):
                processes._spawn_vllm(server, model, str(binary), {})

            cmd = popen.call_args.args[0]
            env = popen.call_args.kwargs["env"]
            self.assertEqual(cmd[:3], [str(binary), "serve", str(model.resolve())])
            self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "fp8_e4m3")
            self.assertEqual(cmd[cmd.index("--attention-backend") + 1], "TRITON_ATTN")
            spec = json.loads(cmd[cmd.index("--speculative-config") + 1])
            self.assertEqual(spec["method"], "mtp")
            self.assertEqual(spec["num_speculative_tokens"], 2)
            self.assertEqual(spec["attention_backend"], "TRITON_ATTN")
            self.assertEqual(cmd[cmd.index("--max-num-seqs") + 1], "4")
            self.assertIn("--enable-prefix-caching", cmd)
            self.assertIn("--enable-auto-tool-choice", cmd)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
            self.assertTrue(server.vllm)

    def test_spawn_vllm_ram_fallback_offloads_kv_cache(self):
        proc = MagicMock(pid=988)
        thread = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Qwen3.8-27B-NVFP4"
            model.mkdir()
            binary = Path(tmp) / "vllm"
            binary.write_text("#!/bin/sh\n")
            server = _server(
                model_path=str(model),
                model_name=model.name,
                alias=model.name,
                format="vllm",
                spill="ram",
                vision=True,
                mtp=True,
            )
            with (
                patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
                patch.object(processes.threading, "Thread", return_value=thread),
            ):
                processes._spawn_vllm(server, model, str(binary), {})

            cmd = popen.call_args.args[0]
            self.assertEqual(
                cmd[cmd.index("--kv-offloading-backend") + 1], "native"
            )
            self.assertEqual(cmd[cmd.index("--kv-offloading-size") + 1], "8")
            self.assertEqual(server.spill, "ram")

    def test_spawn_vllm_128k_uses_long_context_gpu_profile(self):
        proc = MagicMock(pid=989)
        thread = MagicMock()
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "Qwen3.8-27B-NVFP4"
            model.mkdir()
            binary = Path(tmp) / "vllm"
            binary.write_text("#!/bin/sh\n")
            server = _server(
                model_path=str(model),
                model_name=model.name,
                alias=model.name,
                format="vllm",
                ctx=131072,
                spill="none",
                vision=False,
                mtp=True,
            )
            with (
                patch.object(processes.subprocess, "Popen", return_value=proc) as popen,
                patch.object(processes.threading, "Thread", return_value=thread),
            ):
                processes._spawn_vllm(server, model, str(binary), {})

            cmd = popen.call_args.args[0]
            self.assertEqual(
                cmd[cmd.index("--gpu-memory-utilization") + 1], "0.90"
            )
            self.assertEqual(cmd[cmd.index("--max-num-batched-tokens") + 1], "4096")
            self.assertNotIn("--kv-offloading-size", cmd)
            self.assertIn("--language-model-only", cmd)


class AnalyzerTimingTests(unittest.TestCase):
    def tearDown(self):
        analyzer._queries.clear()
        analyzer._live_tps.clear()
        analyzer._slot_state.clear()
        analyzer._history.clear()

    def test_api_timings_match_llama_server_mtp_formula(self):
        # llama-server: predicted_per_second = 1000 / predicted_ms * predicted_n
        # MTP N=2 accepting ~80% → ~112 t/s with 200 tokens in ~1786 ms
        predicted_n = 200
        predicted_ms = 1785.7
        expected = 1000.0 / predicted_ms * predicted_n
        analyzer.note_api_timings(
            "qwen",
            {
                "prompt_n": 128,
                "prompt_ms": 80.0,
                "prompt_per_second": 1600.0,
                "predicted_n": predicted_n,
                "predicted_ms": predicted_ms,
                "predicted_per_second": expected,
                "draft_n": 300,
                "draft_n_accepted": 240,
            },
        )
        q = list(analyzer._queries["qwen"])[-1]
        self.assertAlmostEqual(q["gen_tps"], expected, places=2)
        self.assertEqual(q["draft_n"], 300)
        self.assertEqual(q["draft_n_accepted"], 240)
        self.assertAlmostEqual(analyzer._live_tps["qwen"]["gen_tps"], expected, places=2)

    def test_slot_live_rate_counts_mtp_multi_token_jumps(self):
        # One poll: decoded 10→13 in 0.05s ⇒ 60 t/s (3 accepted drafts)
        analyzer._slot_state["qwen"] = {
            0: {
                "task": 7,
                "processing": True,
                "prompt_proc": 100,
                "prompt_cache": 0,
                "prompt_tok": 100,
                "decoded": 10,
                "t_start": time.time() - 1.0,
                "t_gen_start": time.time() - 0.5,
                "last_decoded": 10,
                "last_ts": time.time() - 0.05,
            }
        }
        now = time.time()
        with patch.object(analyzer.time, "time", return_value=now):
            analyzer._update_from_slots(
                "qwen",
                [
                    {
                        "id": 0,
                        "id_task": 7,
                        "is_processing": True,
                        "n_prompt_tokens_processed": 100,
                        "n_prompt_tokens_cache": 0,
                        "n_prompt_tokens": 100,
                        "next_token": [{"n_decoded": 13}],
                    }
                ],
            )
        self.assertAlmostEqual(analyzer._live_tps["qwen"]["gen_tps"], 60.0, places=1)

    def test_api_timings_preferred_over_slot_estimates(self):
        analyzer.note_api_timings(
            "qwen",
            {
                "prompt_per_second": 950.0,
                "predicted_per_second": 112.0,
                "predicted_n": 64,
                "predicted_ms": 571.0,
            },
        )
        analyzer._queries["qwen"].append(
            {
                "task": 1,
                "ts": time.time(),
                "done": True,
                "source": "slot",
                "prompt_tps": 900.0,
                "gen_tps": 40.0,
            }
        )
        server = _server(id="qwen", status="running", pid=None)
        with (
            patch.object(analyzer, "_fetch_json", return_value=None),
            patch.object(analyzer, "_nvidia_snapshot", return_value=([], [])),
            patch.object(analyzer, "_proc_rss_mib", return_value=None),
        ):
            data = analyzer.analyze_server(server)
        self.assertEqual(data["kpis"]["gen_tps_avg"], 112.0)
        self.assertEqual(data["kpis"]["prompt_tps_avg"], 950.0)

    def test_absurd_tps_spikes_are_rejected(self):
        analyzer.note_api_timings(
            "qwen",
            {
                "prompt_n": 1,
                "prompt_ms": 0.001,
                "prompt_per_second": 230000.0,
                "predicted_n": 1,
                "predicted_ms": 0.001,
                "predicted_per_second": 1000000.0,
            },
        )
        self.assertEqual(list(analyzer._queries["qwen"]), [])
        self.assertNotIn("prompt_tps", analyzer._live_tps["qwen"])
        self.assertNotIn("gen_tps", analyzer._live_tps["qwen"])

        analyzer._history["qwen"].append(
            {"ts": time.time(), "prompt_tps": 230000.0, "gen_tps": 1000000.0}
        )
        server = _server(id="qwen", status="running", pid=None)
        with (
            patch.object(analyzer, "_fetch_json", return_value=None),
            patch.object(analyzer, "_nvidia_snapshot", return_value=([], [])),
            patch.object(analyzer, "_proc_rss_mib", return_value=None),
        ):
            data = analyzer.analyze_server(server)
        self.assertIsNone(data["history"][-1]["prompt_tps"])
        self.assertIsNone(data["history"][-1]["gen_tps"])


class AnalyzerTests(unittest.TestCase):
    def test_nvidia_snapshot_uses_two_queries(self):
        gpu_rows = (
            "0, GPU-a, NVIDIA GeForce RTX 5090, 32607, 30000, 80, 70\n"
            "1, GPU-b, NVIDIA GeForce RTX 3060, 12288, 11000, 5, 2\n"
        )
        app_rows = "42, GPU-a, 25000\n99, GPU-b, 100\n"
        results = [
            SimpleNamespace(returncode=0, stdout=gpu_rows),
            SimpleNamespace(returncode=0, stdout=app_rows),
        ]
        with patch.object(analyzer.subprocess, "run", side_effect=results) as run:
            devices, all_gpus = analyzer._nvidia_snapshot(42)

        self.assertEqual(run.call_count, 2)
        self.assertEqual([d["gpu"] for d in devices], [0])
        self.assertEqual(devices[0]["used_mib"], 25000.0)
        self.assertEqual(len(all_gpus), 2)


if __name__ == "__main__":
    unittest.main()
