"""Offline ranged-download and independent directory-audit regressions."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock

SCRIPTS=Path(__file__).resolve().parent
if not (SCRIPTS/'download-convert-deepseek41-dspark.py').is_file(): SCRIPTS=SCRIPTS.parent
spec=importlib.util.spec_from_file_location('dspark_preparation',SCRIPTS/'download-convert-deepseek41-dspark.py')
subject=importlib.util.module_from_spec(spec);spec.loader.exec_module(subject)


class Response:
    def __init__(self,body,begin,end,total,status=206,wrong_range=False):
        self.body=body;self.reads=0;self.status=status
        self.headers={'Content-Length':str(end-begin+1),'Content-Range':f'bytes {begin}-{end}/{total}' if not wrong_range else 'bytes 0-3/999'}
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,count):
        self.reads+=1;result,self.body=self.body[:count],self.body[count:];return result


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.payload=b'exact-header-and-source-payload-0123456789'
        self.shard={'file':'model-00044.safetensors','file_bytes':len(self.payload),
            'publisher_sha256':hashlib.sha256(self.payload).hexdigest(),'origin_url':'https://example.invalid/pinned-revision/file',
            'header_prefix_bytes':12,'header_prefix_sha256':hashlib.sha256(self.payload[:12]).hexdigest()}
        self.ranges=[]
    def opener(self,request,timeout):
        begin,end=map(int,re.fullmatch(r'bytes=(\d+)-(\d+)',request.get_header('Range')).groups())
        self.ranges.append((begin,end))
        return Response(self.payload[begin:end+1],begin,end,len(self.payload))
    def download(self,opener): return subject.download(self.shard,self.root,lambda event:None,chunk_size=8,opener=opener)

    def test_ignored_range_never_reads_response_body(self):
        response=Response(self.payload,0,7,len(self.payload),status=200)
        with self.assertRaisesRegex(ValueError,'HTTP206'): self.download(lambda *a,**k:response)
        self.assertEqual(response.reads,0)
        self.assertEqual((self.root/(self.shard['file']+'.part')).stat().st_size,0)

    def test_mismatched_content_range_never_reads_body(self):
        response=Response(self.payload,0,7,len(self.payload),wrong_range=True)
        with self.assertRaisesRegex(ValueError,'Content-Range'): self.download(lambda *a,**k:response)
        self.assertEqual(response.reads,0)

    def test_partial_response_resumes_exact_offset_and_verifies_whole_file(self):
        response=Response(self.payload[:4],0,7,len(self.payload))
        with self.assertRaisesRegex(IOError,'Short ranged'): self.download(lambda *a,**k:response)
        self.assertEqual((self.root/(self.shard['file']+'.part')).stat().st_size,4)
        result=self.download(self.opener)
        self.assertEqual(self.ranges[0],(4,11))
        self.assertEqual(result['status'],'downloaded-whole-file-verified')
        self.assertEqual((self.root/self.shard['file']).read_bytes(),self.payload)

    def test_changed_resume_identity_refuses_before_network(self):
        part=self.root/(self.shard['file']+'.part');part.write_bytes(b'old')
        opener=mock.Mock()
        with self.assertRaisesRegex(ValueError,'resume identity'): self.download(opener)
        opener.assert_not_called()
        self.assertEqual(part.read_bytes(),b'old')

    def test_wrong_full_hash_never_publishes_final(self):
        self.shard['publisher_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'publisher'): self.download(self.opener)
        self.assertFalse((self.root/self.shard['file']).exists())
        self.assertEqual((self.root/(self.shard['file']+'.part')).stat().st_size,len(self.payload))

    def test_package_drift_is_rejected_before_execution(self):
        payload=self.root/'script.py';payload.write_bytes(b'pass\n')
        manifest=self.root/'manifest.json'
        manifest.write_text(json.dumps({'files':{'script.py':{'sha256':subject.digest(payload)}}}))
        expected=subject.digest(manifest)
        subject.verify_package(self.root,expected)
        payload.write_bytes(b'changed\n')
        with self.assertRaisesRegex(ValueError,'source drift'): subject.verify_package(self.root,expected)

    def tiny_gguf(self,kind=39,offset=0):
        string=lambda value:struct.pack('<Q',len(value.encode()))+value.encode()
        name='mtp.0.ffn_gate_exps.weight'
        metadata={'general.architecture':'deepseek41-dspark'}
        kv=string('general.architecture')+struct.pack('<I',8)+string('deepseek41-dspark')
        info=string(name)+struct.pack('<I3QIQ',3,32,1,1,kind,offset)
        raw=struct.pack('<IIQQ',0x46554747,3,1,1)+kv+info
        raw+=b'\0'*((-len(raw))%32)
        size=subject.FORMATS[kind][1]
        raw+=b'\0'*size+b'\0'*((-size)%32)
        path=self.root/'tiny.gguf';path.write_bytes(raw)
        expected={'metadata':metadata,'tensors':{name:{'shape':[32,1,1],'type':39,'bytes':17}}}
        return path,expected

    def test_independent_gguf_directory_audit_accepts_exact_geometry(self):
        path,expected=self.tiny_gguf()
        audit=subject.audit_output(path,expected)
        self.assertEqual(audit['status'],'passed-directory-audit')
        self.assertFalse(audit['release_qualified'])

    def test_relabelled_expert_encoding_fails_independent_audit(self):
        path,expected=self.tiny_gguf(kind=8)
        with self.assertRaisesRegex(ValueError,'tensor mismatch'): subject.audit_output(path,expected)

    def test_payload_offset_or_truncation_fails_audit(self):
        path,expected=self.tiny_gguf(offset=32)
        with self.assertRaisesRegex(ValueError,'gap/overlap'): subject.audit_output(path,expected)
        path,expected=self.tiny_gguf();path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError,'file size'): subject.audit_output(path,expected)

    def test_non_linux_execution_refuses_before_resource_process(self):
        with mock.patch.object(sys,'platform','win32'),mock.patch.object(subject.subprocess,'run') as launch:
            with self.assertRaisesRegex(RuntimeError,'VM/Linux only'): subject.quiet_resources(0)
            launch.assert_not_called()

    def process_fixture(self):
        folder=self.root/'51';folder.mkdir()
        fields=['S']+['0']*19;fields[19]='123456'
        (folder/'stat').write_text('51 (nginx worker) '+' '.join(fields)+'\n')
        (folder/'status').write_text('Uid:\t65534\t65534\t65534\t65534\nGid:\t65534\t65534\t65534\t65534\n')
        (folder/'cmdline').write_bytes(b'nginx\0worker\0')
        (folder/'maps').write_text('0000-1000 r-xp 0 0:0 0 /usr/sbin/nginx\n')
        return folder

    def test_direct_maps_reader_keeps_process_identity(self):
        folder=self.process_fixture()
        identity,maps,method=subject.read_process_maps(folder)
        self.assertEqual(identity['start_ticks'],123456)
        self.assertEqual(identity['uids'],[65534]*4)
        self.assertEqual(method,'direct')
        self.assertIn('/usr/sbin/nginx',maps)

    def test_maps_reader_detects_identity_change_during_read(self):
        folder=self.process_fixture();expected=subject.process_identity(folder)
        changed={**expected,'start_ticks':123457}
        with mock.patch.object(subject,'process_identity',side_effect=[expected,changed]):
            with self.assertRaisesRegex(RuntimeError,'changed after'):subject.maps_for_identity(folder,expected)

    def test_root_fallback_drops_to_observed_ids_and_binds_script_bytes(self):
        folder=self.process_fixture();expected=subject.process_identity(folder)
        completed=mock.Mock(returncode=0,stdout=json.dumps({'maps':'/usr/sbin/nginx\n'}),stderr='')
        with mock.patch.object(subject,'maps_for_identity',side_effect=PermissionError),mock.patch.object(subject.os,'geteuid',return_value=0,create=True),mock.patch.object(subject.subprocess,'run',return_value=completed) as run:
            identity,maps,method=subject.read_process_maps(folder)
        argv=run.call_args.args[0];kwargs=run.call_args.kwargs
        self.assertEqual(kwargs['user'],65534);self.assertEqual(kwargs['group'],65534);self.assertEqual(kwargs['extra_groups'],[])
        self.assertEqual(argv[1:3],['-I','-B'])
        self.assertEqual(argv[6],subject.digest(Path(subject.__file__).resolve()))
        self.assertEqual(json.loads(argv[8]),expected)
        self.assertEqual(method,'same-effective-uid-gid-subprocess')
        self.assertEqual(identity,expected)

    def test_nonroot_permission_failure_never_launches_credential_reader(self):
        folder=self.process_fixture()
        with mock.patch.object(subject,'maps_for_identity',side_effect=PermissionError),mock.patch.object(subject.os,'geteuid',return_value=1000,create=True),mock.patch.object(subject.subprocess,'run') as run:
            with self.assertRaises(PermissionError):subject.read_process_maps(folder)
            run.assert_not_called()

    def test_failed_credential_reader_remains_unreadable(self):
        folder=self.process_fixture()
        completed=mock.Mock(returncode=1,stdout='',stderr='Permission denied')
        with mock.patch.object(subject,'maps_for_identity',side_effect=PermissionError),mock.patch.object(subject.os,'geteuid',return_value=0,create=True),mock.patch.object(subject.subprocess,'run',return_value=completed):
            with self.assertRaisesRegex(PermissionError,'Credential-scoped'):subject.read_process_maps(folder)

    def test_pid_reuse_during_credential_reader_is_rejected(self):
        folder=self.process_fixture();expected=subject.process_identity(folder)
        completed=mock.Mock(returncode=0,stdout=json.dumps({'maps':'/usr/sbin/nginx'}),stderr='')
        with mock.patch.object(subject,'process_identity',side_effect=[expected,{**expected,'start_ticks':9}]),mock.patch.object(subject,'maps_for_identity',side_effect=PermissionError),mock.patch.object(subject.os,'geteuid',return_value=0,create=True),mock.patch.object(subject.subprocess,'run',return_value=completed):
            with self.assertRaisesRegex(RuntimeError,'identity changed during'):subject.read_process_maps(folder)

    def test_child_program_executes_only_exact_source_bytes(self):
        folder=self.process_fixture();expected=subject.process_identity(folder)
        argv=['-c',str(Path(subject.__file__).resolve()),subject.digest(Path(subject.__file__).resolve()),str(folder),json.dumps(expected)]
        # Run the same child body locally; privilege dropping is asserted above.
        import io
        from contextlib import redirect_stdout
        with mock.patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()) as capture:
            exec(compile(subject._MAP_READER,'map-reader','exec'),{})
        self.assertIn('/usr/sbin/nginx',json.loads(capture.getvalue())['maps'])
        argv[2]='0'*64
        with mock.patch.object(sys,'argv',argv):
            with self.assertRaisesRegex(RuntimeError,'source changed'):exec(compile(subject._MAP_READER,'map-reader','exec'),{})

    def cgroup_fixture(self,membership,mount):
        proc=self.root/'proc';(proc/'self').mkdir(parents=True)
        (proc/'self/cgroup').write_text(membership+'\n');(proc/'self/mountinfo').write_text(mount+'\n')
        fs=self.root/'fs';fs.mkdir()
        return proc,fs

    def memory_files(self,folder,version,limit,current,stats):
        folder.mkdir(parents=True,exist_ok=True)
        (folder/('memory.limit_in_bytes' if version=='v1' else 'memory.max')).write_text(str(limit))
        (folder/('memory.usage_in_bytes' if version=='v1' else 'memory.current')).write_text(str(current))
        (folder/'memory.stat').write_text(stats)

    def test_actual_v1_container_mount_uses_total_inactive_file(self):
        container='/docker/d71188ba1e2ed8db6edb4cb5b3630ed701777cb55c7728c0b2986d4684740d0e'
        proc,fs=self.cgroup_fixture('9:memory:'+container,
            '1164 1155 0:42 '+container+' /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime master:23 - cgroup cgroup rw,memory')
        self.memory_files(fs/'sys/fs/cgroup/memory','v1',350,340,'inactive_file 3\ntotal_inactive_file 100\n')
        available,observations=subject.cgroup_memory_available(1000,proc,fs)
        self.assertEqual(available,110)
        self.assertEqual(observations[0]['version'],'v1')
        self.assertEqual(observations[0]['inactive_file'],100)

    def test_v1_nested_ancestor_limit_is_not_ignored(self):
        proc,fs=self.cgroup_fixture('9:memory:/parent/child','1 0 0:1 /parent /cg rw - cgroup cgroup rw,memory')
        self.memory_files(fs/'cg/child','v1',1000,10,'total_inactive_file 0\n')
        self.memory_files(fs/'cg','v1',100,90,'total_inactive_file 5\n')
        available,observations=subject.cgroup_memory_available(5000,proc,fs)
        self.assertEqual(available,15);self.assertEqual(len(observations),2)

    def test_v2_nested_mapping_and_unlimited_ancestor(self):
        proc,fs=self.cgroup_fixture('0::/parent/child','1 0 0:1 /parent /cg rw - cgroup2 cgroup rw')
        self.memory_files(fs/'cg/child','v2',100,90,'inactive_file 3\n')
        self.memory_files(fs/'cg','v2','max',200,'inactive_file 50\n')
        available,observations=subject.cgroup_memory_available(5000,proc,fs)
        self.assertEqual(available,13);self.assertIsNone(observations[1]['conservative_available'])

    def test_v1_unlimited_and_namespace_root_preserve_host_cap(self):
        proc,fs=self.cgroup_fixture('9:memory:/','1 0 0:1 /docker/container /cg rw - cgroup cgroup rw,memory')
        self.memory_files(fs/'cg','v1',9223372036854771712,100,'total_inactive_file 10\n')
        available,observations=subject.cgroup_memory_available(123,proc,fs)
        self.assertEqual(available,123);self.assertIsNone(observations[0]['conservative_available'])

    def test_unresolvable_or_missing_controller_fails_closed(self):
        proc,fs=self.cgroup_fixture('9:memory:/other/container','1 0 0:1 /docker/container /cg rw - cgroup cgroup rw,memory')
        with self.assertRaisesRegex(RuntimeError,'Cannot bind'):subject.cgroup_memory_available(1000,proc,fs)
        (proc/'self/mountinfo').write_text('1 0 0:1 / /sys rw - sysfs sysfs rw\n')
        with self.assertRaisesRegex(RuntimeError,'mount is not visible'):subject.cgroup_memory_available(1000,proc,fs)

    def test_hybrid_prefers_explicit_v1_memory_over_unmounted_unified_hierarchy(self):
        proc,fs=self.cgroup_fixture('9:memory:/docker/container\n0::/docker/container',
            '1 0 0:1 /docker/container /cg rw - cgroup cgroup rw,memory')
        self.memory_files(fs/'cg','v1',100,90,'total_inactive_file 3\n')
        available,observations=subject.cgroup_memory_available(1000,proc,fs)
        self.assertEqual(available,13)
        self.assertEqual([r['version'] for r in observations],['v1'])

    def test_hybrid_missing_memory_mount_fails_even_when_unified_is_listed(self):
        proc,fs=self.cgroup_fixture('9:memory:/docker/container\n0::/docker/container',
            '1 0 0:1 / /sys rw - sysfs sysfs rw')
        with self.assertRaisesRegex(RuntimeError,'mount is not visible: v1'):
            subject.cgroup_memory_available(1000,proc,fs)


if __name__=='__main__': unittest.main()
