"""Local-only review UI server."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_head(self):
        # Python 3.9's SimpleHTTPRequestHandler does not implement byte ranges.
        # Browsers use them for video metadata and seeking, so support ranges
        # for local media while retaining the stdlib's safe path translation.
        path = Path(self.translate_path(self.path))
        if not path.is_file() or not str(path.resolve()).startswith(str(ROOT.resolve()) + '/'):
            return super().send_head()
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        range_header = self.headers.get('Range')
        if range_header:
            try:
                unit, span = range_header.split('=', 1)
                if unit != 'bytes' or ',' in span:
                    raise ValueError
                left, right = span.split('-', 1)
                if left:
                    start = int(left)
                    end = min(int(right), size - 1) if right else size - 1
                else:
                    length = int(right)
                    start = max(0, size - length)
                if start < 0 or start >= size or end < start:
                    raise ValueError
                status = 206
            except (ValueError, TypeError):
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return None
        file = path.open('rb')
        if start:
            file.seek(start)
        self.send_response(status)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Content-Length', str(end - start + 1))
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        return file

    def do_GET(self):
        if self.path == '/api/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, 'root': str(ROOT)}).encode())
            return
        if self.path == '/api/tracknet':
            manifest = json.loads((DATA / 'manifest.json').read_text())
            ann_path = DATA / 'annotations.json'
            annotations = json.loads(ann_path.read_text()) if ann_path.exists() else {}
            labels_path = DATA / 'tracknet_labels.json'
            labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
            rows = [r for r in manifest if annotations.get(r['id'], {}).get('usable') is True
                    and 'glove_pre_time' in annotations[r['id']]]
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'rows': rows, 'annotations': annotations, 'labels': labels}).encode())
            return
        if self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            manifest = json.loads((DATA / 'manifest.json').read_text())
            ann_path = DATA / 'annotations.json'
            annotations = json.loads(ann_path.read_text()) if ann_path.exists() else {}
            auto = {}
            for row in manifest:
                path = DATA / 'analysis' / row['id'] / 'auto.json'
                if path.exists():
                    auto[row['id']] = json.loads(path.read_text())
            self.wfile.write(json.dumps({'manifest': manifest, 'annotations': annotations, 'auto': auto}).encode())
            return
        super().do_GET()

    def do_POST(self):
        size = int(self.headers.get('Content-Length', '0'))
        payload = json.loads(self.rfile.read(size))
        if self.path == '/api/cotracker-track':
            try:
                sample_id = payload['sample_id']
                manifest = json.loads((DATA / 'manifest.json').read_text())
                row = next(r for r in manifest if r['id'] == sample_id)
                labels_path = DATA / 'tracknet_labels.json'
                labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
                keyframes = payload.get('keyframes', [])
                from cotracker_assist import track_points
                candidates, metadata = track_points(
                    ROOT / row['video'], payload['start_time'], payload['end_time'],
                    keyframes, sample_id,
                )
                split = 'test' if sample_id in {'FF_03', 'FF_09', 'CU_03', 'SL_03', 'SL_08'} else 'train'
                for key, candidate in candidates.items():
                    old = labels.get(key)
                    if old and old.get('label_source') not in {'interpolated', 'cotracker3'}:
                        continue  # Never replace a human label or keyframe.
                    labels[key] = {
                        **candidate,
                        'pitch_type': row['pitch_type'],
                        'game_pk': row['game_pk'],
                        'source_video': row['video'],
                        'split': split,
                    }
                temporary = labels_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(labels, ensure_ascii=False, indent=2))
                temporary.replace(labels_path)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'labels': {k: labels[k] for k in candidates if k in labels},
                    'metadata': metadata,
                    'candidate_count': len(candidates),
                }).encode())
                return
            except Exception as exc:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(exc)}).encode())
                return
        if self.path == '/api/tracknet-labels':
            target = DATA / 'tracknet_labels.json'
        elif self.path == '/api/annotations':
            target = DATA / 'annotations.json'
        else:
            self.send_error(404)
            return
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        temporary.replace(target)
        self.send_response(204)
        self.end_headers()

    def log_message(self, fmt, *args):
        print(fmt % args)


if __name__ == '__main__':
    print('Review UI: http://127.0.0.1:8765/review.html')
    print('TrackNet labels: http://127.0.0.1:8765/tracknet_label.html')
    ThreadingHTTPServer(('127.0.0.1', 8765), Handler).serve_forever()
