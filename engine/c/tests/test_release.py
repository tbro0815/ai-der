"""Release contract checks, runnable without CUDA or third-party Python packages."""
from pathlib import Path
import os
import runpy
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class ReleaseTests(unittest.TestCase):
    def test_version(self):
        version = runpy.run_path(str(ROOT / 'engine/c/version.py'))['__version__']
        self.assertRegex(version, r'^\d+\.\d+\.\d+$')
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'engine'))
        import sys
        result = subprocess.check_output([sys.executable, '-m', 'colibri.cli', '--version'],
                                         env=env, text=True).strip()
        self.assertEqual(result, f'ai-der {version}')
        package = subprocess.check_output([sys.executable, '-c',
            'from colibri import __version__; print(__version__)'], env=env, text=True).strip()
        self.assertEqual(package, version)

    def test_installer_scripts_parse_and_help(self):
        for name in ('install.sh', 'engine/c/install_models.sh', 'engine/c/install_llamacpp.sh',
                     'engine/c/install_vllm.sh', 'engine/c/write_env.sh'):
            subprocess.run(['bash', '-n', str(ROOT / name)], check=True)
            self.assertTrue(os.access(ROOT / name, os.X_OK), name)
        result = subprocess.run(['bash', str(ROOT / 'install.sh'), '--help'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        for flag in ('--check', '--no-apt', '--core-only', '--skip-models', '--skip-llamacpp',
                     '--skip-vllm', '--systemd', '--data-dir'):
            self.assertIn(flag, result.stdout)

    def test_installer_preflight(self):
        script = (ROOT / 'install.sh').read_text()
        subprocess.run(['bash', '-n', str(ROOT / 'install.sh')], check=True)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'os-release').write_text('ID=ubuntu\nVERSION_ID=24.04\n')
            (base / 'meminfo').write_text('MemTotal:       131000000 kB\n')
            for name, body in {
                'uname': 'if [ "$1" = -s ]; then echo Linux; else echo x86_64; fi',
                'nvidia-smi': 'echo "NVIDIA GeForce RTX 3090 Ti"',
                'nvcc': 'echo "Cuda compilation tools, release 12.4, V12.4.131"',
                'sudo': 'echo "Unexpected mutation" >&2; exit 99',
            }.items():
                p = base / name
                p.write_text('#!/bin/sh\n' + body + '\n')
                p.chmod(0o755)
            (base / 'bin').mkdir()
            (base / 'bin/nvcc').write_bytes((base / 'nvcc').read_bytes())
            (base / 'bin/nvcc').chmod(0o755)
            script = script.replace('/etc/os-release', str(base / 'os-release')).replace(
                '/proc/meminfo', str(base / 'meminfo'))
            test_script = base / 'install.sh'
            test_script.write_text(script)
            env = dict(os.environ, PATH=f'{base}:{os.environ["PATH"]}', CUDA_HOME=tmp)
            def check(code, fragment):
                result = subprocess.run(['bash', str(test_script), '--check'], env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                self.assertIn(fragment, result.stdout + result.stderr)
            check(0, 'CUDA: 12.4')
            check(0, 'Data directory')
            result = subprocess.run(['bash', str(test_script), '--check', '--core-only', '--data-dir', tmp],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('about 5 GB needed', result.stdout)
            (base / 'meminfo').write_text('MemTotal: 64000000 kB\n')
            check(1, '128 GB')
            (base / 'meminfo').write_text('MemTotal: 131000000 kB\n')
            (base / 'bin/nvcc').write_text('#!/bin/sh\necho "release 12.3,"\n')
            check(1, 'CUDA Toolkit 12.4+')
            (base / 'os-release').write_text('ID=ubuntu\nVERSION_ID=22.04\n')
            check(1, 'Ubuntu 24.04')


if __name__ == '__main__':
    unittest.main()
