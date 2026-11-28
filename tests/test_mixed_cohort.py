"""Cohort discovery must not silently impose the old pilot subject limit."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from strokefill3d.mixed_rf.data import cohort_summary

spec=importlib.util.spec_from_file_location('prepare_cohort',Path(__file__).resolve().parents[1]/'scripts/prepare_mixed_cohort.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class CohortTests(unittest.TestCase):
    def test_all_labeled_subjects_and_raters_are_discovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(15):
                d=root/'epi/subjects'/f'sub-{i:04d}'/'postop';d.mkdir(parents=True)
                for name in ['t1mri-1','seg-1','seg-2']:(d/(name+'.nii.gz')).touch()
            cases,excluded=module.discover(SimpleNamespace(episurg=root/'epi',epibios=root/'bio',arc=root/'arc'))
            self.assertEqual(len(cases),15)
            self.assertTrue(all(len(c['labels'])==2 for c in cases))
            self.assertEqual(excluded,[])

    def test_unfinished_preparation_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'dataset.json').write_text(json.dumps({'complete':True}))
            for split in ['train','val','test']:
                with (root/(split+'.csv')).open('w') as f:f.write('domain\natlas\n')
            with self.assertRaisesRegex(ValueError,'incomplete or changed'):
                cohort_summary(root)


if __name__=='__main__':unittest.main()
