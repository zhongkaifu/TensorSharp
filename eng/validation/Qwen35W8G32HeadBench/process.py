"""Own new-session process lifecycle; no GPU or external actions at import."""
import os,signal,subprocess
def command(argv,timeout,stdout=None):
 """Own new-session group only; registration is inside cleanup scope, child unmasks."""
 proc=None;oldmask=None;handlers={};f=stdout
 def interrupted(s,frame):raise KeyboardInterrupt('signal '+str(s))
 try:
  for sig in(signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupted)
  oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM})
  proc=subprocess.Popen(argv,stdout=f or subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask))
  signal.pthread_sigmask(signal.SIG_SETMASK,oldmask);oldmask=None
  if f:return proc.wait(timeout=timeout),''
  text=proc.communicate(timeout=timeout)[0];return proc.returncode,text
 except BaseException:
  if proc is not None and proc.returncode is None:
   for sig in(signal.SIGINT,signal.SIGTERM):signal.signal(sig,signal.SIG_IGN)
   try:
    if os.getpgid(proc.pid)==proc.pid:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=5)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   except ProcessLookupError:proc.wait()
  raise
 finally:
  if oldmask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
  for sig,handler in handlers.items():signal.signal(sig,handler)
