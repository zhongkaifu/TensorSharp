from pathlib import Path
import subprocess,shlex,sys
variant="compensated4"
root=Path("/workspace/ts-dspark-sum-variants-20260916")
repo=Path("/workspace/ts-codex-20260916-r3/repo")
build=repo/"TensorSharp.GGML.Native/build"
with (root/"compensated4-benchmark-build.log").open("w") as log:
 def run(args):
  log.write(shlex.join(args)+"\n");log.flush();subprocess.run(args,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
 run(["/usr/local/cuda/bin/nvcc","-O3","-DNDEBUG","-std=c++17","-arch=sm_86","-extended-lambda","--ftz=false","--prec-div=true","--prec-sqrt=true","-Xcompiler=-fPIC","-Dtsg_matmul_cuda_state=tsg_matmul_cuda_state_candidate","-Dtsg_matmul_cuda_init=tsg_matmul_cuda_init_candidate","-Dtsg_matmul_cuda_free=tsg_matmul_cuda_free_candidate","-Dtsg_matmul_cuda_compute=tsg_matmul_cuda_compute_candidate","-DTSG_GGML_TEST_HOOKS","-DTSG_GGML_USE_CUDA=1","-I"+str(repo/"TensorSharp.GGML.Native"),"-I"+str(repo/"ExternalProjects/ggml/src"),"-I"+str(repo/"ExternalProjects/ggml/include"),"-I/usr/local/cuda/include","-c",str(root/(variant+".cu")),"-o",str(root/"compensated4-candidate-bench.o")])
 run(["/usr/local/cuda/bin/nvcc","-x","cu","-arch=sm_86","-extended-lambda","-O3","-DNDEBUG","-std=c++17","-DTSG_GGML_USE_CUDA=1","-I"+str(repo/"TensorSharp.GGML.Native"),"-I"+str(repo/"TensorSharp.GGML.Native/tests"),"-I"+str(repo/"ExternalProjects/ggml/src"),"-I"+str(repo/"ExternalProjects/ggml/include"),"-I/usr/local/cuda/include","-c",str(root/"benchmark.cpp"),"-o",str(root/"benchmark.o")])
 link=shlex.split((build/"CMakeFiles/GgmlOpsCudaMatmulPrecisionTest.dir/link.txt").read_text())
 old="CMakeFiles/GgmlOpsCudaMatmulPrecisionTest.dir/tests/cuda_matmul_precision_test.cpp.o"
 assert old in link
 link=[str(root/"benchmark.o") if x==old else x for x in link]
 link.insert(1,str(root/"compensated4-candidate-bench.o"))
 link[link.index("-o")+1]=str(root/"compensated4-benchmark")
 run(link)
