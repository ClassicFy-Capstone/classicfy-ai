"""Batch ATEPP YouTube synchronization: run, resume, inventory, benchmark, estimate."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('PYTHONIOENCODING','utf-8')

import argparse
from collections import Counter
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'/'preprocessing'))
from youtube_sync import (VERSION, ELIGIBLE, FS, RATE, atomic_json, candidate_reason,
    configuration, digest, empty_match, execute_job, load_inventory, read_json,
    request_fingerprint, reusable_result, sha, video_id)

WORKSPACE = HERE.parents[2]
DEFAULT_INPUT = WORKSPACE/'outputs/atepp_representative_v2_all_candidates'
DEFAULT_OUTPUT = WORKSPACE/'outputs/atepp_youtube_full'
DEFAULT_BENCHMARK = WORKSPACE/'outputs/atepp_youtube_sync_benchmark/benchmark.json'


@contextmanager
def runner_lock(output):
    """OS lock automatically releases after a crash; no stale PID/unlock deletion."""
    path = Path(output)/'.runner.lock'
    stream = path.open('a+b')
    stream.seek(0,2)
    if stream.tell()==0:
        stream.write(b' ');stream.flush()
    stream.seek(0)
    try:
        if os.name=='nt':
            import msvcrt
            msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('Another runner is using this output directory.')
    try:
        yield
    finally:
        stream.seek(0)
        if os.name=='nt':
            msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            fcntl.flock(stream,fcntl.LOCK_UN)
        stream.close()


def inventory(args):
    source,segments,metadata,groups = load_inventory(args.input,args.midi_root,args.include_excluded)
    output = Path(args.output).resolve()
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('Output and input directories must be separate, non-nested directories.')
    output.mkdir(parents=True,exist_ok=True)
    return source,segments,metadata,groups,output


def make_jobs(args,metadata,groups,output):
    config = configuration(vars(args))
    video_counts = Counter(r.get('youtube_links','') for r in metadata.values())
    options = {k:getattr(args,k) for k in ['save_clips','keep_audio','offline','audio_cache','download_timeout']}
    selected = set(args.ids or [])
    jobs = []
    for pid,segments in groups.items():
        if selected and pid not in selected:
            continue
        row = metadata[pid]
        jobs.append(dict(metadata=row,segments=segments,options=options,config=config,
            output=str(output),shared_video=video_counts[row.get('youtube_links','')]>1,
            fingerprint=request_fingerprint(row,segments,config)))
    if selected-set(groups):
        raise ValueError(f'Unknown or ineligible IDs: {sorted(selected-set(groups))}')
    if args.command=='benchmark' and not selected:
        pool=sorted(jobs,key=lambda j:float(j['metadata']['track_duration']))
        size=min(args.limit or 6,len(pool))
        positions=[round((.1+.89*i/(size-1))*(len(pool)-1)) for i in range(size)] if size>1 else [len(pool)//2]
        jobs=[pool[i] for i in dict.fromkeys(positions)]
    elif args.limit:
        jobs=jobs[:args.limit]
    return jobs,config


def atomic_csv(path,rows,fields):
    tmp=path.with_suffix('.csv.tmp')
    with tmp.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)
    tmp.replace(path)


def gather(segments,metadata,groups,output,config,include_excluded=False):
    """Export all source rows, including failures, skipped rows and pending work."""
    matches={};perf_counts=Counter();records=[]
    for pid,group in groups.items():
        path=output/'jobs'/pid/'result.json'
        fingerprint=request_fingerprint(metadata[pid],group,config)
        try:
            result=read_json(path)
            if result.get('fingerprint')!=fingerprint:
                continue
            if result['status']=='completed' and not reusable_result(path,fingerprint,config['save_clips'],output):
                continue
            perf_counts[result['status']]+=1
            records.append(result)
            for row in result['rows']:
                matches[(pid,row['candidate_number'])]=row
        except (OSError,ValueError,KeyError,TypeError):
            continue
    rows=[]
    for original in segments:
        skip=candidate_reason(original,include_excluded)
        if skip:
            row=empty_match(original,'skipped_'+skip,url=metadata[original['perf_id']].get('youtube_links',''))
        else:
            row=matches.get((original['perf_id'],original['candidate_number']))
            if row is None:
                row=empty_match(original,'pending',url=metadata[original['perf_id']].get('youtube_links',''))
        rows.append(row)
    return rows,perf_counts,records


def export(segments,metadata,groups,output,config,include_excluded=False):
    rows,perf_counts,records=gather(segments,metadata,groups,output,config,include_excluded)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    atomic_csv(output/'youtube_matches.csv',rows,fields)
    atomic_csv(output/'review_queue.csv',[r for r in rows if r['youtube_match_status']=='review_required'],fields)
    atomic_csv(output/'unmatched.csv',[r for r in rows if r['youtube_match_status'] not in {'matched_auto','review_required'}],fields)
    progress=dict(total_input_candidates=len(rows),eligible_performances=len(groups),
        accounted_performances=sum(perf_counts.values()),pending_performances=len(groups)-sum(perf_counts.values()),
        performance_statuses=dict(perf_counts),candidate_statuses=dict(Counter(r['youtube_match_status'] for r in rows)),
        updated_at_utc=datetime.now(timezone.utc).isoformat(),human_verified=False)
    atomic_json(output/'progress.json',progress)
    return progress,records


def stop_process(proc):
    if proc.poll() is not None:
        return
    if os.name=='nt':
        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()


def supervisor_failure(job,work,status,error,elapsed):
    atomic_json(work/'result.json',dict(perf_id=job['metadata']['perf_id'],fingerprint=job['fingerprint'],
        status=status,error=error,rows=[empty_match(s,status,error,job['metadata'].get('youtube_links','')) for s in job['segments']],
        timings=dict(process_seconds=elapsed),finished_at_utc=datetime.now(timezone.utc).isoformat()))


def run(args):
    source,segments,metadata,groups,output=inventory(args)
    jobs,config=make_jobs(args,metadata,groups,output)
    missing=[m for m in ['synctoolbox','numpy','pandas','pretty_midi','soundfile','yt_dlp','imageio_ffmpeg']
             if importlib.util.find_spec(m) is None]
    if missing:
        raise RuntimeError('Missing dependencies; install requirements-youtube-sync.txt: '+', '.join(missing))
    if not args.offline and not shutil.which('node'):
        print('Note: Node.js was not found; YouTube may require a supported JavaScript runtime.',flush=True)
    with runner_lock(output):
        atomic_json(output/'run_manifest.json',dict(version=VERSION,config=config,input=str(source),
            input_segments_sha256=sha(source/'segments.csv'),input_audit_sha256=sha(source/'input_midi_audit.csv'),
            requested_performances=len(jobs),workers=args.workers,options=vars(args),
            created_at_utc=datetime.now(timezone.utc).isoformat(),python=sys.version,
            platform=platform.platform(),logical_cpus=os.cpu_count(),human_verified=False))
        pending=[];skipped=0
        for job in jobs:
            path=output/'jobs'/job['metadata']['perf_id']/'result.json'
            if reusable_result(path,job['fingerprint'],args.save_clips,output):
                skipped+=1;continue
            if path.exists() and not args.retry_failed:
                try:
                    previous=read_json(path)
                    if previous.get('fingerprint')==job['fingerprint'] and previous.get('status')!='completed':
                        skipped+=1;continue
                except (OSError,ValueError):pass
            pending.append(job)
        print(f'{len(jobs)} selected, {skipped} already accounted, {len(pending)} to run; workers={args.workers}',flush=True)
        progress,_=export(segments,metadata,groups,output,config,args.include_excluded)
        started=time.monotonic();last_start=0.;last_export=started;done=0;blocked_streak=0
        active={};cursor=0;stop_reason='';interrupted=False
        try:
            while active or (cursor<len(pending) and not stop_reason):
                while len(active)<args.workers and cursor<len(pending) and not stop_reason:
                    if time.monotonic()-last_start<args.download_delay:
                        break
                    job=pending[cursor];cursor+=1
                    pid=job['metadata']['perf_id'];work=output/'jobs'/pid;work.mkdir(parents=True,exist_ok=True)
                    if (work/'result.json').exists():
                        (work/'result.json').replace(work/'previous_result.json')
                    atomic_json(work/'request.json',job)
                    log=(work/'worker.log').open('w',encoding='utf-8')
                    kwargs=dict(stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ))
                    if os.name=='nt':kwargs['creationflags']=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.CREATE_NO_WINDOW
                    else:kwargs['start_new_session']=True
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'_worker',str(work/'request.json')],**kwargs)
                    active[pid]=dict(proc=proc,job=job,work=work,log=log,started=time.monotonic())
                    last_start=time.monotonic()
                for pid,item in list(active.items()):
                    proc=item['proc'];elapsed=time.monotonic()-item['started']
                    if proc.poll() is None and elapsed<=args.job_timeout:
                        continue
                    if proc.poll() is None:
                        stop_process(proc)
                        supervisor_failure(item['job'],item['work'],'timeout','job_wall_timeout',elapsed)
                    path=item['work']/'result.json'
                    try:
                        result=read_json(path)
                        if result.get('fingerprint')!=item['job']['fingerprint']:
                            raise ValueError('stale_result')
                    except (OSError,ValueError):
                        supervisor_failure(item['job'],item['work'],'worker_crashed',f'worker_exit_code:{proc.returncode}',elapsed)
                        result=read_json(path)
                    result['timings']['process_seconds']=elapsed
                    atomic_json(path,result)
                    item['log'].close();del active[pid];done+=1
                    blocked_streak=blocked_streak+1 if result['status']=='access_blocked' else 0
                    print(f'[{done}/{len(pending)}] {pid}: {result["status"]} ({elapsed:.1f}s)',flush=True)
                    if blocked_streak>=3:
                        stop_reason='three_consecutive_access_blocks; resume later with --retry-failed'
                        print('Stopping new downloads: '+stop_reason,flush=True)
                if time.monotonic()-last_export>=args.checkpoint_seconds:
                    progress,_=export(segments,metadata,groups,output,config,args.include_excluded)
                    last_export=time.monotonic()
                if active or (cursor<len(pending) and not stop_reason):
                    time.sleep(.2)
        except KeyboardInterrupt:
            interrupted=True;stop_reason='keyboard_interrupt'
            print('Stopping workers; committed results will be reused on the next run.',flush=True)
        finally:
            for item in active.values():
                stop_process(item['proc']);item['log'].close()
            progress,records=export(segments,metadata,groups,output,config,args.include_excluded)
            atomic_json(output/'last_run.json',dict(elapsed_seconds=time.monotonic()-started,
                completed_this_run=done,skipped_existing=skipped,stop_reason=stop_reason,progress=progress))
            if args.command=='benchmark':
                benchmark_records=[]
                wanted={j['metadata']['perf_id'] for j in jobs}
                for record in records:
                    if record['status']=='completed' and record['perf_id'] in wanted:
                        benchmark_records.append(dict(perf_id=record['perf_id'],**record['timings'],
                                                      mapping_bytes=(output/'jobs'/record['perf_id']/'mapping.npz').stat().st_size))
                atomic_json(output/'benchmark.json',dict(workers=args.workers,
                    elapsed_seconds=time.monotonic()-started,records=benchmark_records,
                    manifest=str(output/'run_manifest.json'),source='fresh per-performance worker measurements'))
        print(json.dumps(progress,ensure_ascii=False,indent=2),flush=True)
        failures=sum(count for status,count in progress['performance_statuses'].items() if status!='completed')
        return 130 if interrupted else 2 if failures or stop_reason else 0


def nonnegative_line(samples):
    """Fit seconds = per-file overhead + rate * audio seconds without NumPy."""
    if not samples:
        raise ValueError('A benchmark is required for a measured time estimate.')
    xs,ys=zip(*samples);xm,ym=statistics.mean(xs),statistics.mean(ys)
    denominator=sum((x-xm)**2 for x in xs)
    slope=max(0.,sum((x-xm)*(y-ym) for x,y in samples)/denominator) if denominator else sum(ys)/sum(xs)
    intercept=ym-slope*xm
    if intercept<0:
        intercept=0.;slope=sum(x*y for x,y in samples)/sum(x*x for x in xs)
    return intercept,slope


def estimate(args):
    source,segments,metadata,groups,output=inventory(args)
    durations=[float(metadata[pid]['track_duration']) for pid in groups]
    durations.sort();total=sum(durations);count=len(durations)
    report=dict(input=str(source),performances=count,input_rows=len(segments),
        eligible_candidates=sum(map(len,groups.values())),input_statuses=dict(Counter(s['status'] for s in segments)),
        audio_hours=total/3600,median_track_seconds=statistics.median(durations),
        p90_track_seconds=durations[int(.9*(count-1))],longest_track_seconds=max(durations),
        logical_cpus=os.cpu_count(),workers=args.workers,
        whole_audio_wav_gib=total*FS*2/1024**3,
        candidate_wav_gib=sum(float(r['end_seconds'])-float(r['start_seconds']) for rows in groups.values() for r in rows)*FS*2/1024**3,
        note='All accessible source videos assumed. Unavailable videos produce explicit unmatched rows; full successful coverage cannot be guaranteed.')
    path=Path(args.benchmark_report)
    if path.is_file():
        benchmark=read_json(path);records=benchmark['records']
        # A first cold process may compile/cache scientific kernels. Count this
        # startup once, rather than projecting it onto every one of 11k files.
        model_records=records[1:] if benchmark.get('workers')==1 and len(records)>=5 else records
        cpu_samples=[(r['audio_duration_seconds'],max(0.,r['process_seconds']-r.get('download_seconds',0)-r.get('decode_seconds',0))) for r in model_records]
        overhead,rate=nonnegative_line(cpu_samples)
        downloaded=[r for r in records if r.get('audio_source')=='download']
        if downloaded:
            net_overhead,net_rate=nonnegative_line([(r['audio_duration_seconds'],r['download_seconds']+r.get('decode_seconds',0)) for r in downloaded])
            download_total=net_overhead*count+net_rate*total
        else:
            net_overhead,net_rate=None,None
            download_total=count*8.+total*.015  # Explicit unmeasured network assumption.
        cpu_total=overhead*count+rate*total
        # These are conservative planning ranges, not measured parallel speedup.
        efficiencies={1:(1.,1.),2:(1.45,1.8),3:(1.8,2.5),4:(2.1,3.1)}
        speed_lo,speed_hi=efficiencies.get(args.workers,(max(1.,args.workers*.45),max(1.,args.workers*.7)))
        sequential=cpu_total+download_total
        delay_floor=max(0.,(count-1)*args.download_delay)
        low=max(delay_floor,sequential*.8/speed_hi)
        high=max(delay_floor,sequential*1.6/speed_lo)
        mapping_rate=sum(r.get('mapping_bytes',0) for r in records)/sum(r['audio_duration_seconds'] for r in records)
        report.update(benchmark_report=str(path.resolve()),benchmark_successful_tracks=len(records),
            compute_model_tracks=len(model_records),
            cold_warmup_excluded=records[0]['perf_id'] if len(model_records)<len(records) else None,
            measured_network_tracks=len(downloaded),cpu_model=dict(per_file_seconds=overhead,seconds_per_audio_second=rate),
            network_model=dict(per_file_seconds=net_overhead,seconds_per_audio_second=net_rate,
                fallback_assumption=None if downloaded else '8s/file + 0.015s/audio second; not measured'),
            serial_compute_hours=cpu_total/3600,serial_download_decode_hours=download_total/3600,
            predicted_serial_total_hours=sequential/3600,
            predicted_workers_hours_low=low/3600,predicted_workers_hours_high=high/3600,
            retained_mapping_gib=mapping_rate*total/1024**3,
            assumptions=['No YouTube throttling, CAPTCHA, region blocks or long outages.',
                'Computer remains awake; sustained CPU speed comparable to benchmark.',
                'Parallel efficiency and 0.8–1.6 duration multipliers are planning assumptions, not confidence intervals.',
                'Short benchmark cannot fully predict longest or edited recordings; manual review time excluded.',
                'Default deletes owned full-source media after each attempted job; mapping/results remain.'])
    else:
        report['estimate_status']='Run benchmark first; no invented measured runtime.'
    atomic_json(output/'runtime_estimate.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    worker=sub.add_parser('_worker',help=argparse.SUPPRESS);worker.add_argument('request')
    for name in ['run','benchmark','estimate','status']:
        cmd=sub.add_parser(name)
        cmd.add_argument('--input',default=str(DEFAULT_INPUT))
        cmd.add_argument('--output',default=str(DEFAULT_OUTPUT if name!='benchmark' else WORKSPACE/'outputs/atepp_youtube_sync_benchmark'))
        cmd.add_argument('--midi-root')
        cmd.add_argument('--workers',type=int,default=2)
        cmd.add_argument('--ids',nargs='+')
        cmd.add_argument('--limit',type=int)
        cmd.add_argument('--include-excluded',action='store_true')
        cmd.add_argument('--save-clips',action='store_true')
        cmd.add_argument('--keep-audio',action='store_true')
        cmd.add_argument('--audio-cache')
        cmd.add_argument('--offline',action='store_true')
        cmd.add_argument('--retry-failed',action='store_true')
        cmd.add_argument('--download-delay',type=float,default=2.)
        cmd.add_argument('--download-timeout',type=int,default=600)
        cmd.add_argument('--job-timeout',type=int,default=1800)
        cmd.add_argument('--checkpoint-seconds',type=float,default=300.)
        cmd.add_argument('--benchmark-report',default=str(DEFAULT_BENCHMARK))
    args=parser.parse_args()
    if args.command!='_worker':
        if args.workers<1 or args.workers>8:parser.error('--workers must be between 1 and 8')
        if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
        if args.job_timeout<=0 or args.download_timeout<=0 or args.download_delay<0 or args.checkpoint_seconds<=0:
            parser.error('Timeouts/checkpoints must be positive; delay must be nonnegative')
        if args.command=='benchmark' and not args.ids and not args.limit:
            args.limit=6  # Never accidentally benchmark the whole corpus.
    return args


def main():
    args=parse_args()
    if args.command=='_worker':return execute_job(args.request)
    if args.command=='estimate':return estimate(args)
    if args.command=='status':
        _,segments,metadata,groups,output=inventory(args)
        config=configuration(vars(args))
        if (output/'run_manifest.json').is_file():
            config=read_json(output/'run_manifest.json')['config']
        with runner_lock(output):
            # Status reports the saved run's mode, including optional clip output.
            if config.get('include_excluded') and not args.include_excluded:
                _,segments,metadata,groups=load_inventory(args.input,args.midi_root,True)
            progress,_=export(segments,metadata,groups,output,config,config['include_excluded'])
        print(json.dumps(progress,ensure_ascii=False,indent=2));return 0
    return run(args)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RuntimeError,OSError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(1)
