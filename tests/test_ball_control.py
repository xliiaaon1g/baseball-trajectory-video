import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ball_control_experiment import (DEFAULT_INPUT, validate_inputs, score_rows,
                                    detect_fixture, evaluate_annotation, sha)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tracks=np.load(DEFAULT_INPUT/'original_ff/tracks.npy')
        self.rows=[{'frame':i,'status':'unique','center_px':self.tracks[i,6].astype(float).tolist()}
                   for i in range(81)]

    def test_real_pair_contract(self):
        self.assertGreater(validate_inputs(DEFAULT_INPUT)['changed_ball_grid_samples'],0)

    def test_missing_and_uncertain_are_not_dropped(self):
        rows=copy.deepcopy(self.rows)
        for i in [1,2]:rows[i]={'frame':i,'status':'absent' if i==1 else 'uncertain','center_px':None}
        result=score_rows(rows,self.tracks)
        self.assertEqual(result['scored_frames'],80)
        self.assertEqual(result['path_success_fraction'],78/80)

    def test_background_failure_even_when_path_is_exact(self):
        result=score_rows(self.rows,self.tracks,[True]+[False]*80)
        self.assertEqual(result['path_success_fraction'],1)
        self.assertEqual(result['path_and_scene_success_fraction'],0)

    def test_wrong_clip_and_unreviewed_rejected(self):
        payload={'schema':'blind_ball_annotations_v1','width':832,'height':480,
                 'blind_id':'not_found','method':'human','annotations':self.rows}
        with self.assertRaises(ValueError):evaluate_annotation(payload,[],DEFAULT_INPUT)
        rows=copy.deepcopy(self.rows);rows[7]['status']='unreviewed'
        with self.assertRaises(ValueError):score_rows(rows,self.tracks)

    def test_duplicate_frames_and_nonfinite_centers_rejected(self):
        rows=copy.deepcopy(self.rows);rows[5]['frame']=4
        with self.assertRaises(ValueError):score_rows(rows,self.tracks)
        rows=copy.deepcopy(self.rows);rows[5]['center_px']=[float('nan'),50]
        with self.assertRaises(ValueError):score_rows(rows,self.tracks)

    def test_synthetic_annotations_do_not_become_generated_performance(self):
        payload={'schema':'blind_ball_annotations_v1','width':832,'height':480,
                 'blind_id':'clip_001','method':'synthetic fixture detector','annotations':self.rows}
        mapping=[{'blind_id':'clip_001','condition':'original_ff',
                  'condition_sha256':sha(DEFAULT_INPUT/'original_ff/tracks.npy')}]
        with self.assertRaises(ValueError):evaluate_annotation(payload,mapping,DEFAULT_INPUT)
        self.assertEqual(evaluate_annotation(payload,mapping,DEFAULT_INPUT,True)['path_success_fraction'],1)

    def test_mutated_conditions_rejected(self):
        payload={'schema':'blind_ball_annotations_v1','width':832,'height':480,
                 'blind_id':'clip_001','method':'human','annotations':self.rows}
        mapping=[{'blind_id':'clip_001','condition':'original_ff','condition_sha256':'wrong'}]
        with self.assertRaises(ValueError):evaluate_annotation(payload,mapping,DEFAULT_INPUT)

    def test_detector_does_not_choose_nearest_ball(self):
        image=np.zeros((480,832,3),dtype=np.uint8)
        image[100:105,100:105]=255;image[300:305,500:505]=255
        result=detect_fixture(image)
        self.assertEqual(result['status'],'multiple')
        self.assertIsNone(result['center_px'])


if __name__=='__main__':unittest.main()
