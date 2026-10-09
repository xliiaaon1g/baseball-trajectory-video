"""Reproducible workflow: select, match Savant CSV, download, annotate, export."""
import argparse, collections, concurrent.futures, csv, datetime, hashlib, json, math
import pathlib, subprocess, urllib.request, time

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'data'

def read_json(path):
    return json.loads(path.read_text())

def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temp.replace(path)

def download(url, path):
    if path.exists() and path.stat().st_size > 1000:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            temp = path.with_suffix(path.suffix + '.part')
            with urllib.request.urlopen(request, timeout=45) as source, temp.open('wb') as out:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk: break
                    out.write(chunk)
            temp.replace(path)
            return
        except Exception:
            if attempt == 2: raise
            time.sleep(1 + attempt)

def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,codec_name,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration',
        '-of', 'json', str(path)]))

def prepare():
    candidates = read_json(ROOT.parent/'quick_data_audit/right_pitcher_pilot/bradish_home_FF_CU_SL_413.json')
    counts = collections.defaultdict(collections.Counter)
    for x in candidates:
        counts[x['current']['game_pk']][x['current']['event']['details']['type']['code']] += 1
    # Prefer maximal within-game balance, then maximal same-game count, then latest date.
    primary = max(counts, key=lambda g: (min(counts[g][t] for t in ['FF','CU','SL']), sum(counts[g].values()), g == 716434))
    selected = []
    for pitch in ['FF','CU','SL']:
        group = [x for x in candidates if x['current']['game_pk'] == primary and x['current']['event']['details']['type']['code'] == pitch]
        group.sort(key=lambda x: (x['current']['about']['atBatIndex'], x['current']['event']['pitchNumber']))
        selected.extend(group[:10])
        if len(group) < 10:
            date = datetime.date.fromisoformat(group[0]['source']['date'])
            rest = [x for x in candidates if x['current']['game_pk'] != primary and x['current']['event']['details']['type']['code'] == pitch]
            rest.sort(key=lambda x: (abs((datetime.date.fromisoformat(x['source']['date'])-date).days), x['current']['about']['atBatIndex'], x['current']['event']['pitchNumber']))
            selected.extend(rest[:10-len(group)])
    sources = {pid:'https://sporty-clips.mlb.com/'+media+'.mp4' for pid,media in read_json(ROOT/'media_sources.json')}
    manifest = []
    for i, x in enumerate(selected):
        current=x['current'];event=current['event'];game=current['game_pk']
        csv_path=DATA/'raw'/f'savant_{game}.csv'
        url=f'https://baseballsavant.mlb.com/statcast_search/csv?all=true&type=details&game_pk={game}'
        download(url,csv_path)
        with csv_path.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
        matches=[r for r in rows if int(r['game_pk'])==game and int(r['pitcher'])==680694
                 and int(r['at_bat_number'])==current['about']['atBatIndex']+1 and int(r['pitch_number'])==event['pitchNumber']]
        if len(matches)!=1: raise ValueError(f'{event["playId"]}: Savant join not unique: {len(matches)}')
        stat=matches[0]
        assert stat['p_throws']=='R' and stat['home_team']=='BAL' and stat['inning_topbot']=='Top'
        assert stat['pitch_type']==event['details']['type']['code'] and stat['description'] in ['ball','called_strike']
        assert int(stat['batter'])==current['matchup']['batter']['id']
        assert abs(float(stat['release_speed'])-event['pitchData']['startSpeed'])<0.11
        pid=event['playId'];pitch=stat['pitch_type'];sample_id=f'{pitch}_{sum(r["pitch_type"]==pitch for r in manifest)+1:02d}'
        manifest.append({'id':sample_id,'play_id':pid,'pitch_type':pitch,'date':stat['game_date'],
            'game_pk':game,'primary_game':game==primary,'batter':current['matchup']['batter']['fullName'],
            'savant_page':x['source']['video_link'],'media_url':sources.get(pid),
            'video':f'data/videos/{sample_id}.mp4','savant':stat,'savant_csv':str(csv_path.relative_to(ROOT)),
            'savant_csv_url':url,'mlb_event':event,'join_key':[game,int(stat['at_bat_number']),event['pitchNumber']],
            'matched':True,'video_status':'pending'})
    assert collections.Counter(r['pitch_type'] for r in manifest)=={'FF':10,'CU':10,'SL':10}
    save_json(DATA/'manifest.json',manifest)
    save_json(DATA/'selection.json',{'primary_game':primary,'primary_candidates':dict(counts[primary]),
             'selected_games':dict(collections.Counter(r['game_pk'] for r in manifest)),
             'retrieved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'match_method':'playId -> MLB event -> (game_pk,at_bat_number,pitch_number); pitcher/batter/hand/speed/type assertions'})
    print('Prepared',len(manifest),'Savant-matched rows.',flush=True)

def fetch_videos(workers=3):
    rows=read_json(DATA/'manifest.json')
    def worker(row):
        try:
            if not row['media_url']:raise ValueError('Media URL missing; resolve source page and update media_sources.json')
            path=ROOT/row['video'];download(row['media_url'],path);metadata=probe(path)
            stream=metadata['streams'][0]
            if stream['width']<1280 or stream['height']<720:raise ValueError('Source below required 720p')
            row.update(video_status='downloaded',probe=metadata,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            row.pop('video_error',None)
            print(row['id'],'downloaded',path.stat().st_size,flush=True)
        except Exception as e:row.update(video_status='failed',video_error=str(e));print(row['id'],'FAILED',e,flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(worker,rows))
    save_json(DATA/'manifest.json',rows)
    print(collections.Counter(r['video_status'] for r in rows),flush=True)

def analyze(limit=None):
    from annotate import analyze_video
    rows=read_json(DATA/'manifest.json')
    for row in (rows[:limit] if limit else rows):
        if not row['video_status'].startswith('downloaded'):continue
        out=DATA/'analysis'/row['id'];out.mkdir(parents=True,exist_ok=True)
        result=analyze_video(ROOT/row['video'],row,out)
        save_json(out/'auto.json',result)
        print(row['id'],'auto release / Savant plate reference',result['release_time'],result['plate_time'],result['method'],flush=True)

def export_clips():
    rows=read_json(DATA/'manifest.json');manual=read_json(DATA/'annotations.json') if (DATA/'annotations.json').exists() else {}
    exported=[]
    for row in rows:
        annotation=manual.get(row['id'],{})
        if not annotation.get('reviewed') or not annotation.get('usable') or not annotation.get('no_swing') or not annotation.get('continuous'):continue
        start=max(0,annotation['release_time']-0.25);end=annotation['glove_pre_time']+1/60
        source=ROOT/row['video'];output=DATA/'clips'/f'{row["id"]}.mp4';output.parent.mkdir(exist_ok=True)
        # Decode then encode; don't use keyframe-only stream-copy cutting.
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(source),'-ss',str(start),'-t',str(end-start),
            '-map','0:v:0','-an','-c:v','libx264','-crf','18','-preset','fast','-fps_mode','passthrough',str(output)],check=True)
        exported.append({'id':row['id'],'video':str(output.relative_to(ROOT)),'original_time_offset':start,
            'release_time_in_clip':annotation['release_time']-start,'glove_pre_time_in_clip':annotation['glove_pre_time']-start,
            'annotation':annotation,'savant':row['savant'],'play_id':row['play_id']})
    save_json(DATA/'clips_manifest.json',exported);print('Exported',len(exported),'reviewed clips')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','download','analyze','export'])
    parser.add_argument('--limit',type=int);parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='download':fetch_videos(args.workers)
    elif args.command=='analyze':analyze(args.limit)
    else:export_clips()
