"""CPU-only tiny/sparse fixtures, no model download, GPU or full payload scan."""
import contextlib,copy,importlib.util,io,json,math,os,struct,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('ninfer_inventory',HERE/'ninfer-inspect.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)

def tensor(name='w',fmt='NVFP4',layout='blockscale-k16-m128x4-v1',shape=None,offset=0):
 shape=[128,64]if shape is None else shape;g=M.geometry(fmt,layout,shape)
 return dict(name=name,kind='tensor',format=fmt,layout=layout,shape=shape,offset=offset,bytes=g['expected_bytes'] or 4096)
def directory(rows):return {'identity':{'model_id':'tiny','weights_id':'fixture'},'objects':rows}
def make(path,rows,divisor=1.,extra=0,rawjson=None):
 raw=json.dumps(directory(rows),separators=(',',':')).encode()if rawjson is None else rawjson;start=((16+len(raw)+4095)//4096)*4096;end=max((x['offset']+x['bytes']for x in rows),default=0)
 with path.open('wb')as f:
  f.write(b'NINFER\0\x02'+struct.pack('<Q',len(raw)));f.write(raw);f.truncate(start+end+extra)
  for r in rows:
   if r.get('format')=='NVFP4':f.seek(start+r['offset']+r['bytes']-4);f.write(struct.pack('<f',divisor))
 return start,len(raw)

class Geometry(unittest.TestCase):
 def test_direct_scalars_shapes_and_precise_sizes(self):
  for fmt,width in [('BF16',2),('FP32',4),('I32',4),('I64',8)]:
   self.assertEqual(M.geometry(fmt,'contiguous-le-v1',[])['expected_bytes'],width);self.assertEqual(M.geometry(fmt,'contiguous-le-v1',[2,3,5])['expected_bytes'],30*width)
 def test_grouped_row_split_ragged_and_actual_head_size(self):
  for fmt,size in [('Q4G64_F16S',536),('Q5G64_F16S',792),('Q6G64_F16S',792),('W8G32_F16S',816)]:self.assertEqual(M.geometry(fmt,'row-split-k128-v1',[3,129])['expected_bytes'],size)
  self.assertEqual(M.geometry('W8G32_F16S','row-split-k128-v1',[248320,5120])['expected_bytes'],1350860800)
 def test_NVFP4_and_FP8_exact_geometry(self):
  g=M.geometry('NVFP4','blockscale-k16-m128x4-v1',[128,64]);self.assertEqual((g['code_bytes'],g['scale_offset'],g['scale_bytes'],g['divisor_offset'],g['expected_bytes']),(4096,4096,512,4608,4612));self.assertEqual(M.geometry('FP8_E4M3FN_ROW_BF16S','row-scale-v1',[3,67])['expected_bytes'],262)
 def test_unknown_incompatible_and_ragged_NVFP4_refused(self):
  for fmt,layout,shape in [('FP16','contiguous-le-v1',[1]),('BF16','invented',[1]),('BF16','row-split-k128-v1',[2,128]),('Q4G64_F16S','contiguous-le-v1',[1]),('NVFP4','blockscale-k16-m128x4-v1',[127,64]),('NVFP4','blockscale-k16-m128x4-v1',[128,63]),('BF16','trellis-tile16-v1',[16,64])]:
   with self.subTest(fmt=fmt,layout=layout,shape=shape),self.assertRaises(M.Refusal):M.geometry(fmt,layout,shape)
 def test_integer_type_rank_positive_and_u64_overflows(self):
  for shape in [[True],[1.0],[0],[-1],[1]*17,[M.U64_MAX,2],[1<<64]]:
   with self.subTest(shape=shape),self.assertRaises(M.Refusal):M.geometry('FP32','contiguous-le-v1',shape)
  with self.assertRaises(M.Refusal):M.geometry('W8G32_F16S','row-split-k128-v1',[1,M.U64_MAX])
 def test_known_opaque_is_not_fully_validated(self):
  row=tensor(fmt='TRELLIS_MUL1_K3',layout='trellis-tile16-v1',shape=[16,64]);v=M.validate_directory(directory([row]),8192,4096);self.assertFalse(v['structure_and_geometry_validated']);self.assertEqual(v['unsupported_geometry_names'],['w']);self.assertIsNone(v['objects'][0]['geometry']['expected_bytes'])

class Directory(unittest.TestCase):
 def test_resource_alignment_and_scalar(self):
  rows=[dict(name='tokenizer',kind='resource',encoding='raw-bytes-v1',offset=0,bytes=3),tensor('n','FP32','contiguous-le-v1',[],256)];v=M.validate_directory(directory(rows),8192,4096);self.assertEqual(v['resource_count'],1);self.assertEqual(v['tensor_count'],1)
 def test_closed_schema_and_duplicates_refused(self):
  for mode in ['rootextra','identityextra','objectextra','resourceencoding','nameempty','duplicate']:
   rows=[tensor()];d=directory(rows)
   if mode=='rootextra':d['components']=[]
   elif mode=='identityextra':d['identity']['id']='wrong'
   elif mode=='objectextra':rows[0]['scale']=1
   elif mode=='resourceencoding':d['objects']=[dict(name='r',kind='resource',encoding='json',offset=0,bytes=1)]
   elif mode=='nameempty':rows[0]['name']=''
   else:d['objects'].append(dict(rows[0],offset=4864))
   with self.subTest(mode=mode),self.assertRaises(M.Refusal):M.validate_directory(d,20000,4096)
 def test_offsets_bounds_order_bytes_and_typed_numbers(self):
  for mode in ['overlap','reverse','alignment','beyondEOF','size','booloffset','floatbytes','overflow']:
   a=tensor();b=tensor('b',offset=4864)
   if mode=='overlap':b['offset']=256
   elif mode=='reverse':a,b=b,a
   elif mode=='alignment':b['offset']+=1
   elif mode=='beyondEOF':b['offset']=19968
   elif mode=='size':b['bytes']-=1
   elif mode=='booloffset':b['offset']=True
   elif mode=='floatbytes':b['bytes']=4612.0
   else:b['offset']=M.U64_MAX-255
   with self.subTest(mode=mode),self.assertRaises(M.Refusal):M.validate_directory(directory([a,b]),20000,4096)
 def test_duplicate_json_members_nonfinite_constants_and_invalid_UTF8(self):
  for data in [b'{"identity":{},"identity":{},"objects":[]}',b'{"x":NaN}',b'{"x":Infinity}',b'\xff',b'{']:
   with self.subTest(data=data),self.assertRaises(M.Refusal):M.strict_json(data)
 def test_metadata_only_does_not_claim_divisors_checked(self):
  v=M.validate_directory(directory([tensor()]),12000,4096);self.assertTrue(v['structure_and_geometry_validated']);self.assertFalse(v['structure_geometry_and_divisors_validated']);self.assertEqual(v['divisor_count_not_read'],1)
 def test_object_count_limit(self):
  with self.assertRaises(M.Refusal)as e:M.validate_directory(directory([tensor()]),12000,4096,max_objects=0)
  self.assertEqual(e.exception.status,'INSPECTION_LIMIT')

class FileInspect(unittest.TestCase):
 def test_valid_fullfile_and_operator_SHA_is_not_recomputed(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'tiny.ninfer';start,size=make(p,[tensor()],divisor=2.);v=M.inspect(p,known_sha256='a'*64);self.assertEqual(v['inspection_status'],'VALIDATED_V2_STRUCTURE_AND_DIVISORS');self.assertFalse(v['full_artifact_sha256_recomputed']);self.assertFalse(v['import_compatible']);self.assertEqual(v['inventory']['objects'][0]['weight_divisor'],2.);self.assertEqual(v['read_statistics']['bytes'],16+size+4)
 def test_nonpositive_and_nonfinite_divisor_rejected(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'tiny'
   for value in [0.,-0.,-1.,float('nan'),float('inf'),-float('inf')]:
    make(p,[tensor()],divisor=value)
    with self.subTest(value=value),self.assertRaises(M.Refusal):M.inspect(p)
 def test_nonregular_artifact_refused_before_read(self):
  with tempfile.TemporaryDirectory()as t:
   with self.assertRaises(M.Refusal):M.inspect(Path(t))
 def test_identified_v3_VI_and_v1_do_not_validate_schema(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'file'
   for magic in [b'NINFER\0\x03',b'VELOXVI\x01',b'NINFER\0\x01']:
    p.write_bytes(magic);v=M.inspect(p);self.assertEqual(v['inspection_status'],'UNSUPPORTED_SCHEMA');self.assertFalse(v['schema_validated']);self.assertEqual(v['read_statistics']['bytes'],8)
   p.write_bytes(b'GGUFxxxx');self.assertEqual(M.inspect(p)['inspection_status'],'UNSUPPORTED_MAGIC')
 def test_truncation_prefix_metadata_and_limit_refused(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'file'
   for data in [b'NINFER\0\x02',b'NINFER\0\x02'+struct.pack('<Q',0),b'NINFER\0\x02'+struct.pack('<Q',M.U64_MAX),b'NINFER\0\x02'+struct.pack('<Q',4096)+b'{}']:
    p.write_bytes(data)
    with self.subTest(data=data),self.assertRaises(M.Refusal):M.inspect(p)
   make(p,[tensor()])
   with self.assertRaises(M.Refusal)as e:M.inspect(p,max_json_bytes=1)
   self.assertEqual(e.exception.status,'INSPECTION_LIMIT')
 def test_sparse_payload_not_read_or_hashed(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'sparse';_,size=make(p,[tensor()],extra=1<<30);v=M.inspect(p);self.assertGreater(v['file_bytes'],1<<30);self.assertEqual(v['read_statistics']['bytes'],16+size+4);self.assertEqual(v['read_statistics']['divisor_reads'],1)
 def test_unknown_geometry_remains_unsupported_not_PASS(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'file';make(p,[tensor(fmt='TRELLIS_MUL1_K2',layout='trellis-tile16-v1',shape=[16,64])]);v=M.inspect(p);self.assertEqual(v['inspection_status'],'UNSUPPORTED_TENSOR_GEOMETRY');self.assertFalse(v['import_compatible'])
 def test_CLI_error_and_unsupported_never_exit_success(self):
  with tempfile.TemporaryDirectory()as t:
   p=Path(t)/'f';p.write_bytes(b'NINFER\0\x03')
   with contextlib.redirect_stdout(io.StringIO())as out:self.assertEqual(M.main([str(p)]),2)
   self.assertEqual(json.loads(out.getvalue())['inspection_status'],'UNSUPPORTED_SCHEMA');p.write_bytes(b'NINFER\0\x02')
   with contextlib.redirect_stdout(io.StringIO())as out:self.assertEqual(M.main([str(p)]),1)
   self.assertFalse(json.loads(out.getvalue())['schema_validated'])
 def test_output_existing_and_dangling_alias_are_not_modified(self):
  with tempfile.TemporaryDirectory()as t:
   root=Path(t).resolve()/'artifacts';root.mkdir();p=root/'out.json';p.write_text('existing')
   with self.assertRaises(M.Refusal):M.write_output(p,{})
   self.assertEqual(p.read_text(),'existing');q=root/'link';q.symlink_to(root/'missing')
   with self.assertRaises(M.Refusal):M.write_output(q,{})
   self.assertFalse((root/'missing').exists());M.write_output(root/'fresh.json',{'ok':True});self.assertEqual(json.loads((root/'fresh.json').read_text()),{'ok':True})

if __name__=='__main__':unittest.main()
