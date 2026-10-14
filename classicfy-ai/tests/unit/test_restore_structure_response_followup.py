import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

script = Path(__file__).resolve().parents[2] / 'scripts/restore_structure_response_followup.py'
spec = importlib.util.spec_from_file_location('restore_followup', script)
restore_followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore_followup)


class RestoreFollowupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'archive'; self.archive.mkdir()
        self.artifacts = self.root / 'artifacts'; self.artifacts.mkdir()
        self.repo = self.root / 'repo'; self.repo.mkdir()
        self.datasets = self.root / 'datasets'; self.datasets.mkdir()
        self.entries = []
        for relative, content, storage in (('response_model/model.py', b'archived source', 'git_snapshot'),
                                           ('response_model/examples.npz', b'external artifact', 'external_artifact')):
            base = self.archive if storage == 'git_snapshot' else self.artifacts
            path = base / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
            self.entries.append(dict(path=relative, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), storage=storage))
        self.write_manifest()

    def write_manifest(self):
        (self.archive / 'archive_manifest.json').write_text(json.dumps({'files': self.entries}))

    def test_verified_restore_preserves_sources_and_expected_workspace_names(self):
        plan = restore_followup.plan_restore(self.archive, self.artifacts)
        out = self.root / 'replay'
        restore_followup.restore(plan, out, self.repo, self.datasets)
        self.assertEqual((out / 'capstone-2026-2-classicfy').resolve(), self.repo.resolve())
        self.assertEqual((out / 'datasets').resolve(), self.datasets.resolve())
        for source, relative in plan:
            self.assertEqual((out / 'diagnostics' / relative).read_bytes(), source.read_bytes())

    def test_check_plan_does_not_write_to_source_or_create_output(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        restore_followup.plan_restore(self.archive, self.artifacts)
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))

    def test_existing_output_is_refused_before_mutation(self):
        plan = restore_followup.plan_restore(self.archive, self.artifacts)
        out = self.root / 'replay'; out.mkdir(); marker = out / 'keep'; marker.write_text('original')
        with self.assertRaises(FileExistsError):
            restore_followup.restore(plan, out, self.repo, self.datasets)
        self.assertEqual(marker.read_text(), 'original')
        self.assertEqual(list(out.iterdir()), [marker])

    def test_changed_external_bytes_are_refused(self):
        (self.artifacts / 'response_model/examples.npz').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            restore_followup.plan_restore(self.archive, self.artifacts)

    def test_missing_artifact_is_reported(self):
        (self.artifacts / 'response_model/examples.npz').unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'Missing'):
            restore_followup.plan_restore(self.archive, self.artifacts)

    def test_traversal_and_duplicate_paths_are_refused(self):
        self.entries[0]['path'] = '../outside'; self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Invalid archive path'):
            restore_followup.plan_restore(self.archive, self.artifacts)
        self.entries[0]['path'] = 'response_model/model.py'
        self.entries.append(dict(self.entries[0])); self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            restore_followup.plan_restore(self.archive, self.artifacts)


if __name__ == '__main__':
    unittest.main(verbosity=2)
