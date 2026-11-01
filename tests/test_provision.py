"""Verify portable private configuration without real GPU or credentials."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/provision.py'


class ProvisionTests(unittest.TestCase):
    def test_paths_and_scope_follow_arguments(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = [sys.executable, str(SCRIPT), '--directory', str(root / 'config'),
                    '--model-path', str(root / 'base'), '--data-root', str(root / 'data'),
                    '--model-root', str(root / 'models'), '--llamafactory-root', str(root / 'engine'),
                    '--project', 'test-project', '--actor', 'test-user']
            result = subprocess.run(args, capture_output=True, text=True, check=True)
            config = json.loads((root / 'config/config.json').read_text())
            self.assertEqual(config['python'], sys.executable)
            self.assertEqual(config['data_root'], str((root / 'data').resolve()))
            self.assertEqual(config['model_root'], str((root / 'models').resolve()))
            self.assertEqual(config['models'][0]['path'], str((root / 'base').resolve()))
            self.assertEqual(config['llamafactory_root'], str((root / 'engine').resolve()))
            service = json.loads((root / 'config/credentials.json').read_text())['services'][0]
            self.assertEqual(service['projects'], ['test-project'])
            self.assertEqual(service['actors'], ['test-user'])
            secret = (root / 'config/service.key').read_text().strip()
            self.assertNotIn(secret, result.stdout + result.stderr)

    def test_existing_credentials_are_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            args = [sys.executable, str(SCRIPT), '--directory', folder]
            subprocess.run(args, capture_output=True, check=True)
            before = (Path(folder) / 'service.key').read_bytes()
            second = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual((Path(folder) / 'service.key').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
