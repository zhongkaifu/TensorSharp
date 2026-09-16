from pathlib import Path
import subprocess,shlex,sys
variant=sys.argv[1]
root=Path("/workspace/ts-dspark-sum-variants-20260916")
repo=Path("/workspace/ts-codex-20260916-r3/repo")
build=repo/"TensorSharp.GGML.Native/build"
with (root/(variant+"-build.log")).open("w") as log:
 def run(args):
  log.write(shlex.join(args)+"\n");log.flush();subprocess.run(args,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
 run(["/usr/local/cuda/bin/nvcc","-O3","-DNDEBUG","-std=c++17","-arch=sm_86","-extended-lambda","--ftz=false","--prec-div=true","--prec-sqrt=true","-Xcompiler=-fPIC","-DTSG_GGML_TEST_HOOKS","-DTSG_GGML_USE_CUDA=1","-I"+str(repo/"TensorSharp.GGML.Native"),"-I"+str(repo/"ExternalProjects/ggml/src"),"-I"+str(repo/"ExternalProjects/ggml/include"),"-I/usr/local/cuda/include","-c",str(root/(variant+".cu")),"-o",str(root/(variant+".o"))])
 link=shlex.split((build/"CMakeFiles/GgmlOps.dir/link.txt").read_text())
 old="CMakeFiles/GgmlOps.dir/ggml_ops_matmul_precision.cu.o"
 assert old in link
 link=[str(root/(variant+".o")) if x==old else x for x in link]
 link[link.index("-o")+1]=str(root/(variant+".so"))
 run(link)
