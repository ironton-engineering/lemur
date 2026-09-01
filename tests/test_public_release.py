import os
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import system_probe
from server import gpu as gpu_module

ROOT = Path(__file__).resolve().parents[1]


class SystemProbeTests(unittest.TestCase):
    def test_supported_gpu_generations(self):
        for name, capability in (
            ("NVIDIA GeForce RTX 3060", "8.6"),
            ("NVIDIA GeForce RTX 4090", "8.9"),
            ("NVIDIA GeForce RTX 5090", "12.0"),
        ):
            with self.subTest(name=name):
                self.assertTrue(system_probe.gpu_supported(name, capability)[0])

    def test_old_and_non_rtx_gpus_are_rejected(self):
        self.assertFalse(
            system_probe.gpu_supported("NVIDIA GeForce RTX 2080 Ti", "7.5")[0]
        )
        self.assertFalse(system_probe.gpu_supported("NVIDIA A100", "8.0")[0])

    def test_mixed_gpu_output_separates_supported_and_ignored_devices(self):
        rows = system_probe.parse_nvidia_smi(
            "0, NVIDIA GeForce RTX 4090, 24564, 8.9, 575.60\n"
            "1, NVIDIA GeForce RTX 2080 Ti, 11264, 7.5, 575.60\n"
        )
        self.assertTrue(rows[0].supported)
        self.assertFalse(rows[1].supported)
        self.assertEqual(system_probe.cuda_architectures(rows), ["89"])

    def test_cuda_architectures_are_unique_cmake_values(self):
        rows = system_probe.parse_nvidia_smi(
            "0, NVIDIA GeForce RTX 3090, 24576, 8.6, 575.60\n"
            "1, NVIDIA GeForce RTX 4090, 24564, 8.9, 575.60\n"
            "2, NVIDIA GeForce RTX 5090, 32607, 12.0, 575.60\n"
            "3, NVIDIA GeForce RTX 4090, 24564, 8.9, 575.60\n"
        )
        self.assertEqual(system_probe.cuda_architectures(rows), ["86", "89", "120"])

    def test_inspect_system_accepts_supported_ubuntu(self):
        with tempfile.TemporaryDirectory() as temp:
            os_release = Path(temp) / "os-release"
            os_release.write_text('ID=ubuntu\nVERSION_ID="24.04"\n')
            gpu = system_probe.GPUReport(
                0, "NVIDIA GeForce RTX 4090", 24564, "8.9", "575.60", True, "supported"
            )
            with (
                patch.object(system_probe, "query_gpus", return_value=([gpu], None)),
                patch.object(system_probe, "disk_free_gib", return_value=100.0),
                patch.object(system_probe.platform, "machine", return_value="x86_64"),
            ):
                report = system_probe.inspect_system(os_release_path=os_release)
        self.assertTrue(report["ok"])

    def test_inspect_system_accepts_ubuntu_26_with_driver_580_or_newer(self):
        with tempfile.TemporaryDirectory() as temp:
            os_release = Path(temp) / "os-release"
            os_release.write_text('ID=ubuntu\nVERSION_ID="26.04"\n')
            gpu = system_probe.GPUReport(
                0, "NVIDIA GeForce RTX 5090", 32607, "12.0", "595.84", True, "supported"
            )
            with (
                patch.object(system_probe, "query_gpus", return_value=([gpu], None)),
                patch.object(system_probe, "disk_free_gib", return_value=100.0),
                patch.object(system_probe.platform, "machine", return_value="x86_64"),
            ):
                report = system_probe.inspect_system(os_release_path=os_release)
        self.assertTrue(report["ok"])
        self.assertEqual(report["minimum_driver"], "580.0")

    def test_inspect_system_rejects_old_driver_on_ubuntu_26(self):
        with tempfile.TemporaryDirectory() as temp:
            os_release = Path(temp) / "os-release"
            os_release.write_text('ID=ubuntu\nVERSION_ID="26.04"\n')
            gpu = system_probe.GPUReport(
                0, "NVIDIA GeForce RTX 5090", 32607, "12.0", "579.99", True, "supported"
            )
            with (
                patch.object(system_probe, "query_gpus", return_value=([gpu], None)),
                patch.object(system_probe, "disk_free_gib", return_value=100.0),
                patch.object(system_probe.platform, "machine", return_value="x86_64"),
            ):
                report = system_probe.inspect_system(os_release_path=os_release)
        self.assertFalse(report["ok"])
        self.assertIn("NVIDIA driver 580.0 or newer is required", report["errors"])

    def test_inspect_system_rejects_low_disk_and_old_driver(self):
        with tempfile.TemporaryDirectory() as temp:
            os_release = Path(temp) / "os-release"
            os_release.write_text('ID=ubuntu\nVERSION_ID="22.04"\n')
            gpu = system_probe.GPUReport(
                0, "NVIDIA GeForce RTX 3090", 24576, "8.6", "550.10", True, "supported"
            )
            with (
                patch.object(system_probe, "query_gpus", return_value=([gpu], None)),
                patch.object(system_probe, "disk_free_gib", return_value=4.0),
                patch.object(system_probe.platform, "machine", return_value="x86_64"),
            ):
                report = system_probe.inspect_system(os_release_path=os_release)
        self.assertFalse(report["ok"])
        self.assertIn("NVIDIA driver 570.26 or newer is required", report["errors"])
        self.assertIn("At least 15 GiB of free disk space is required", report["errors"])

    def test_runtime_gpu_parser_exposes_driver_and_compute_capability(self):
        result = subprocess.CompletedProcess(
            [],
            0,
            stdout="0, NVIDIA GeForce RTX 4070, 12282, 1024, 11258, 8.9, 575.60\n",
            stderr="",
        )
        with patch.object(gpu_module.subprocess, "run", return_value=result):
            found = gpu_module.list_gpus()
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].compute_capability, "8.9")
        self.assertEqual(found[0].driver_version, "575.60")
        self.assertTrue(found[0].supported)


class InstallerLayoutTests(unittest.TestCase):
    def test_test_mode_install_is_repeatable_and_uninstall_keeps_user_data(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            home = temp_path / "home"
            home.mkdir()
            install_root = home / ".local/share/llm-hub"
            bin_dir = home / ".local/bin"
            config_dir = home / ".config/lemur"
            config_dir.mkdir(parents=True)
            state = config_dir / "state.json"
            state.write_text('{"settings": {}}\n')
            model = home / "models/model.gguf"
            model.parent.mkdir()
            model.write_bytes(b"GGUF")
            env = {
                **os.environ,
                "HOME": str(home),
                "LEMUR_INSTALL_ROOT": str(install_root),
                "LEMUR_BIN_DIR": str(bin_dir),
                "LEMUR_TEST_MODE": "1",
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local/share"),
            }
            command = [str(ROOT / "scripts/install-release.sh"), "--no-desktop", "--yes"]
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            self.assertTrue((install_root / "current").is_symlink())
            self.assertTrue((install_root / "releases/0.1.0").is_dir())
            self.assertTrue((bin_dir / "lemur").is_symlink())
            subprocess.run(
                [str(ROOT / "scripts/uninstall.sh"), "--yes"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(install_root.exists())
            self.assertTrue(state.exists())
            self.assertTrue(model.exists())

    def test_command_help_and_version(self):
        help_result = subprocess.run(
            [str(ROOT / "scripts/lemur"), "help"],
            check=True,
            capture_output=True,
            text=True,
        )
        version_result = subprocess.run(
            [str(ROOT / "scripts/lemur"), "version"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("doctor", help_result.stdout)
        self.assertEqual(version_result.stdout.strip(), "Lemur 0.1.0")

    def test_release_launcher_uses_relocatable_virtual_environment_command(self):
        launcher = (ROOT / "scripts/run.sh").read_text()
        self.assertIn('"$VENV/bin/python" -m uvicorn', launcher)
        self.assertNotIn('"$VENV/bin/uvicorn" server.main:app', launcher)

    def test_release_window_has_no_development_browser_controls(self):
        window = (ROOT / "scripts/window.py").read_text()
        self.assertNotIn("set_enable_developer_extras(True)", window)
        self.assertNotIn("KEY_F5", window)

    def test_vllm_installer_uses_a_checked_relocatable_environment(self):
        installer = (ROOT / "scripts/install-vllm.sh").read_text()
        versions = (ROOT / "release/versions.env").read_text()
        self.assertIn("--relocatable", installer)
        self.assertIn("UV_X86_64_LINUX_SHA256", installer)
        self.assertIn("--index-strategy unsafe-best-match", installer)
        self.assertIn("VLLM_PYTHON_VERSION=3.10", versions)
        self.assertIn('"$backend/bin/vllm" --version', installer)
        self.assertIn('f"vllm @ {wheel_uri} \\\\"', installer)
        self.assertIn('settings["vllm_bin"] = sys.argv[2]', installer)
        self.assertIn("at least 20 GiB free", installer)
        self.assertIn("vllm-*.dist-info/licenses/LICENSE", installer)
        self.assertNotIn('cp "$ROOT/LICENSE" "$stage/licenses/Apache-2.0-LICENSE"', installer)

    def test_rollback_swaps_current_and_previous(self):
        with tempfile.TemporaryDirectory() as temp:
            install_root = Path(temp) / "home/.local/share/llm-hub"
            first = install_root / "releases/0.1.0"
            second = install_root / "releases/0.2.0"
            first.mkdir(parents=True)
            second.mkdir()
            (install_root / "current").symlink_to(second)
            (install_root / "previous").symlink_to(first)
            env = {**os.environ, "LEMUR_INSTALL_ROOT": str(install_root)}
            subprocess.run(
                [str(ROOT / "scripts/lemur"), "rollback"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual((install_root / "current").resolve(), first)
            self.assertEqual((install_root / "previous").resolve(), second)

    def test_uninstall_can_remove_user_data_but_not_models(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "home"
            install_root = home / ".local/share/llm-hub"
            config_dir = home / ".config/lemur"
            model = home / "models/model.gguf"
            install_root.mkdir(parents=True)
            config_dir.mkdir(parents=True)
            model.parent.mkdir()
            model.write_bytes(b"GGUF")
            env = {
                **os.environ,
                "HOME": str(home),
                "LEMUR_INSTALL_ROOT": str(install_root),
                "LEMUR_BIN_DIR": str(home / ".local/bin"),
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local/share"),
            }
            subprocess.run(
                [
                    str(ROOT / "scripts/uninstall.sh"),
                    "--yes",
                    "--remove-user-data",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertFalse(config_dir.exists())
            self.assertTrue(model.exists())

    def test_desktop_installer_uses_standard_user_directories(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "home"
            env = {
                **os.environ,
                "HOME": str(home),
                "LEMUR_BIN_DIR": str(home / ".local/bin"),
                "XDG_DATA_HOME": str(home / ".local/share"),
            }
            subprocess.run(
                [str(ROOT / "scripts/install-desktop.sh")],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            desktop = home / ".local/share/applications/lemur.desktop"
            self.assertTrue(desktop.is_file())
            self.assertIn(str(home / ".local/bin/lemur"), desktop.read_text())
            self.assertFalse((home / "Desktop/Lemur.desktop").exists())

    def test_bootstrap_rejects_a_bad_archive_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            fake_bin = base / "bin"
            fake_bin.mkdir()
            archive = base / "archive"
            archive.write_bytes(b"not a release")
            checksum = base / "checksum"
            checksum.write_text("0" * 64 + "  lemur-linux-x86_64.tar.gz\n")
            manifest = base / "manifest"
            manifest.write_text(
                '{"archive":"lemur-linux-x86_64.tar.gz","sha256":"'
                + "0" * 64
                + '","size":13}\n'
            )
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                "#!/usr/bin/env bash\n"
                "set -eu\n"
                "out=''\nurl=''\n"
                "while (($#)); do\n"
                "  case \"$1\" in\n"
                "    --output) out=$2; shift 2;;\n"
                "    http*) url=$1; shift;;\n"
                "    *) shift;;\n"
                "  esac\n"
                "done\n"
                "case \"$url\" in\n"
                "  *.sha256) cp \"$FAKE_CHECKSUM\" \"$out\";;\n"
                "  */manifest.json) cp \"$FAKE_MANIFEST\" \"$out\";;\n"
                "  *) cp \"$FAKE_ARCHIVE\" \"$out\";;\n"
                "esac\n"
            )
            fake_curl.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                "FAKE_ARCHIVE": str(archive),
                "FAKE_CHECKSUM": str(checksum),
                "FAKE_MANIFEST": str(manifest),
                "LEMUR_RELEASE_URL": "https://example.invalid",
            }
            result = subprocess.run(
                [str(ROOT / "install.sh")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum does not match", result.stderr)

    def test_bootstrap_reports_network_loss_and_cleans_temporary_files(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            fake_bin = base / "bin"
            fake_tmp = base / "tmp"
            fake_bin.mkdir()
            fake_tmp.mkdir()
            fake_curl = fake_bin / "curl"
            fake_curl.write_text("#!/bin/sh\nexit 22\n")
            fake_curl.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                "TMPDIR": str(fake_tmp),
                "LEMUR_RELEASE_URL": "https://example.invalid",
            }
            result = subprocess.run(
                [str(ROOT / "install.sh")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(list(fake_tmp.iterdir()), [])

    def test_noninteractive_install_stops_when_packages_are_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            home = base / "home"
            fake_bin = base / "bin"
            home.mkdir()
            fake_bin.mkdir()
            fake_dpkg = fake_bin / "dpkg-query"
            fake_dpkg.write_text("#!/bin/sh\nexit 1\n")
            fake_dpkg.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(home),
                "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                "LEMUR_INSTALL_ROOT": str(home / ".local/share/llm-hub"),
                "LEMUR_BIN_DIR": str(home / ".local/bin"),
                "LEMUR_SKIP_SYSTEM_CHECK": "1",
                "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_DATA_HOME": str(home / ".local/share"),
            }
            result = subprocess.run(
                [
                    str(ROOT / "scripts/install-release.sh"),
                    "--no-desktop",
                    "--non-interactive",
                ],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Approval is required", result.stderr)
            self.assertIn("sudo apt-get install", result.stdout)
            self.assertFalse((home / ".local/share/llm-hub").exists())

    def test_update_network_failure_preserves_active_release(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            install_root = base / "home/.local/share/llm-hub"
            active = install_root / "releases/0.1.0"
            active.mkdir(parents=True)
            (install_root / "current").symlink_to(active)
            fake_bin = base / "bin"
            fake_bin.mkdir()
            fake_curl = fake_bin / "curl"
            fake_curl.write_text("#!/bin/sh\nexit 22\n")
            fake_curl.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                "LEMUR_INSTALL_ROOT": str(install_root),
                "LEMUR_RELEASE_URL": "https://example.invalid",
            }
            result = subprocess.run(
                [str(ROOT / "scripts/lemur"), "update"],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual((install_root / "current").resolve(), active)

    def test_successful_update_keeps_the_old_release_for_rollback(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            install_root = base / "home/.local/share/llm-hub"
            old_release = install_root / "releases/0.1.0"
            new_release = install_root / "releases/0.2.0"
            old_release.mkdir(parents=True)
            (install_root / "current").symlink_to(old_release)
            fake_bin = base / "bin"
            fake_bin.mkdir()
            fake_installer = base / "install.sh"
            fake_installer.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "mkdir -p \"$LEMUR_INSTALL_ROOT/releases/0.2.0\"\n"
                "ln -sfn \"$(readlink -f \"$LEMUR_INSTALL_ROOT/current\")\" "
                "\"$LEMUR_INSTALL_ROOT/previous\"\n"
                "ln -sfn \"$LEMUR_INSTALL_ROOT/releases/0.2.0\" "
                "\"$LEMUR_INSTALL_ROOT/current\"\n"
            )
            fake_curl = fake_bin / "curl"
            fake_curl.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "while (($#)); do\n"
                "  if [[ \"$1\" == --output ]]; then cp \"$FAKE_INSTALLER\" \"$2\"; exit; fi\n"
                "  shift\n"
                "done\n"
                "exit 2\n"
            )
            fake_curl.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
                "FAKE_INSTALLER": str(fake_installer),
                "LEMUR_INSTALL_ROOT": str(install_root),
                "LEMUR_RELEASE_URL": "https://example.invalid",
            }
            subprocess.run(
                [str(ROOT / "scripts/lemur"), "update", "--non-interactive"],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual((install_root / "current").resolve(), new_release)
            self.assertEqual((install_root / "previous").resolve(), old_release)

    def test_optional_vllm_driver_failure_keeps_lemur(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp) / "home"
            install_root = home / ".local/share/llm-hub"
            current = install_root / "current"
            current.mkdir(parents=True)
            fake_bin = Path(temp) / "bin"
            fake_bin.mkdir()
            fake_smi = fake_bin / "nvidia-smi"
            fake_smi.write_text("#!/bin/sh\necho 550.10\n")
            fake_smi.chmod(0o755)
            env = {
                **os.environ,
                "HOME": str(home),
                "LEMUR_INSTALL_ROOT": str(install_root),
                "PATH": str(fake_bin) + os.pathsep + os.environ["PATH"],
            }
            result = subprocess.run(
                [str(ROOT / "scripts/install-vllm.sh")],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(current.exists())


class ReleaseArchiveTests(unittest.TestCase):
    def test_release_builder_makes_a_checked_clean_archive(self):
        subprocess.run(
            [str(ROOT / "release/build-release.sh")],
            check=True,
            capture_output=True,
            text=True,
        )
        archive = ROOT / "dist/lemur-linux-x86_64.tar.gz"
        checksum_file = ROOT / "dist/lemur-linux-x86_64.tar.gz.sha256"
        expected = checksum_file.read_text().split()[0]
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual(actual, expected)
        manifest = json.loads((ROOT / "dist/manifest.json").read_text())
        self.assertEqual(manifest["archive"], archive.name)
        self.assertEqual(manifest["sha256"], actual)
        self.assertEqual(manifest["size"], archive.stat().st_size)
        self.assertTrue(manifest["archive_url"].startswith("https://"))
        listing = subprocess.run(
            ["tar", "-tzf", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for forbidden in ("/.git/", "/.venv/", "/.planning/", "/.cursor/", "__pycache__"):
            self.assertNotIn(forbidden, listing)


if __name__ == "__main__":
    unittest.main()
