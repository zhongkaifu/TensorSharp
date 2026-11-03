#!/usr/bin/env python3
"""Portable real managed-CLI contracts. No real model file, HF, CUDA or network."""
import argparse,json,pathlib,subprocess,tempfile

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dll',type=pathlib.Path,required=True);p.add_argument('--dotnet',default='dotnet');a=p.parse_args();dll=a.dll.resolve();checks=[]
    def run(name,args,code=None):
        r=subprocess.run([a.dotnet,str(dll),*map(str,args)],capture_output=True,text=True,timeout=30)
        assert (r.returncode==0 if code==0 else r.returncode!=0 if code else True),(name,r.stdout,r.stderr)
        checks.append(dict(name=name,exit=r.returncode));return r
    plan=json.loads(run('default-plan',[],0).stdout);assert plan['planOnly']and plan['complete']is False and plan['GPU']=='NOT_RUN'
    run('help',['--help'],0)
    actual=json.loads(run('CPU-inverse-calibration-frame',['--self-test'],0).stdout);assert actual['self_test_pass']and actual['CPU_assertions']==26
    paths=json.loads(run('CPU-output-paths',['--self-test-output'],0).stdout);assert paths['output_contract_pass']and paths['CPU_path_contracts']==6
    run('unknown',['--foreign'],1);run('duplicate',['--execute','--execute'],1);run('missing-value',['--artifact'],1);run('invalid-sha',['--directory-sha','g'*64],1);run('missing-required',['--execute'],1)
    # Fixture is deliberately invalid: tests only failure persistence and output ownership,
    # never pretends a synthetic file is a positive real-frontend/906 binding fixture.
    with tempfile.TemporaryDirectory(prefix='import-cli-')as temporary:
        root=pathlib.Path(temporary).resolve();artifacts=root/'artifacts';artifacts.mkdir();tiny=root/'invalid.ninfer';tiny.write_bytes(b'invalid metadata')
        def args(out):return ['--execute','--artifact',tiny,'--directory-sha','a'*64,'--out',out]
        foreign=artifacts/'foreign';foreign.mkdir();sentinel=foreign/'result.json';sentinel.write_text('foreign-sentinel')
        run('existing-output-preserved',args(foreign),1);assert sentinel.read_text()=='foreign-sentinel'
        alias=artifacts/'alias';alias.symlink_to(foreign,target_is_directory=True)
        run('alias-parent-refused',args(alias/'child'),1);assert not(foreign/'child').exists()and sentinel.read_text()=='foreign-sentinel'
        dangling=artifacts/'dangling';dangling.symlink_to(root/'missing',target_is_directory=True);run('dangling-output-refused',args(dangling),1);assert not(root/'missing').exists()
        run('nonignored-output-refused',args(root/'nonignored'),1);assert not(root/'nonignored').exists()
        own=artifacts/'own';run('owned-reader-error-preserved',args(own),1);error=json.loads((own/'result.json').read_text());assert error['complete']is False and error['GPU']=='NOT_RUN'and error['inference']=='NOT_IMPLEMENTED'
        canonical=artifacts/'canonical-owned';run('canonical-path-used-for-check-and-write',args(alias/'..'/canonical.name),1);assert(canonical/'result.json').is_file()and not(root/'canonical-owned').exists()and sentinel.read_text()=='foreign-sentinel'
    print(json.dumps(dict(cli_contracts_pass=True,actual_CLI_calls=len(checks),checks=checks,independent_helper_CPU_assertions=actual['CPU_assertions'],output_path_contracts=paths['CPU_path_contracts'],GPU='NOT_RUN',actual_artifact_public_frontend='NOT_RUN',HF_oracle='NOT_RUN'),indent=2))

if __name__=='__main__':main()
