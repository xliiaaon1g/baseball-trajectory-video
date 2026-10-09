"""Budget protection: shutdown shell invocation and timeout child cleanup."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('retry', Path(__file__).resolve().parents[1]/'autodl_ff_sl_retry.py')
retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retry)


class Controls(unittest.TestCase):
    def test_shutdown_uses_shell_and_is_bounded(self):
        with tempfile.TemporaryDirectory() as td, patch.object(retry,'ROOT',Path(td)), patch.object(retry.os,'sync'), patch.object(retry.subprocess,'run') as run:
            retry.shutdown()
            self.assertEqual(run.call_args.args[0],['/bin/bash','/usr/bin/shutdown'])
            self.assertTrue(run.call_args.kwargs['check'])
            self.assertEqual(run.call_args.kwargs['timeout'],45)

    def test_timeout_stops_spawned_process_group(self):
        import os,sys
        with tempfile.TemporaryDirectory() as td, patch.object(retry,'ROOT',Path(td)):
            retry.deadline=time.time()+60
            heartbeat=Path(td)/'child_heartbeat'
            child="import time; from pathlib import Path; p=Path(%r); i=0\nwhile True: p.write_text(str(i)); i+=1; time.sleep(.02)" % str(heartbeat)
            code="import subprocess,time,sys; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(60)" % child
            with self.assertRaises(subprocess.TimeoutExpired):
                retry.run('child_cleanup',[sys.executable,'-c',code],limit=0.5)
            count=heartbeat.read_text()
            time.sleep(.15)
            self.assertEqual(heartbeat.read_text(),count,'child continued running after timeout')

    def test_expired_deadline_does_not_start_work(self):
        with tempfile.TemporaryDirectory() as td, patch.object(retry,'ROOT',Path(td)), patch.object(retry.subprocess,'Popen') as popen:
            retry.deadline=time.time()-1
            with self.assertRaises(TimeoutError): retry.run('expired',['unused'])
            popen.assert_not_called()


if __name__=='__main__': unittest.main()
