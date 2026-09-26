import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import engines


def tensorsharp_command(cpu_moe=None, **kwargs):
    """The server command line TensorSharpServer would spawn for these options."""
    spec = engines.config.BackendSpec(
        backend_id="test_moe", display="test", kind="gpu", ts_backend="ggml_cuda")
    model = SimpleNamespace(gguf=Path("/tmp/model.gguf"), mmproj=None,
                            is_diffusion=False)
    server = engines.TensorSharpServer(model, "test_moe", Path("/tmp/unused.log"),
                                       cpu_moe=cpu_moe, **kwargs)
    with patch.object(engines, "_port_open", return_value=False), \
         patch.object(engines.config, "BACKENDS", {"test_moe": spec}), \
         patch.object(engines.config, "TENSORSHARP_SERVER_DLL", Path("/tmp/server.dll")), \
         patch.object(engines.config, "tp_device_env", return_value={}), \
         patch.object(server, "_spawn") as spawn:
        server.start()
    return spawn.call_args[0][0]


def llama_command(cpu_moe=None, **kwargs):
    """The same, for llama-server."""
    spec = engines.config.BackendSpec(
        backend_id="test_moe", display="test", kind="gpu", llama_ngl=999)
    model = SimpleNamespace(gguf=Path("/tmp/model.gguf"), mmproj=None)
    server = engines.LlamaCppServer(model, "test_moe", Path("/tmp/unused.log"),
                                    cpu_moe=cpu_moe, **kwargs)
    with patch.object(engines, "_port_open", return_value=False), \
         patch.object(engines.config, "BACKENDS", {"test_moe": spec}), \
         patch.object(engines.config, "llama_server_exe_for",
                      return_value=Path("/tmp/llama-server")), \
         patch.object(engines.config, "tp_device_env", return_value={}), \
         patch.object(server, "_spawn") as spawn:
        server.start()
    return spawn.call_args[0][0]


class BackendLaunchTests(unittest.TestCase):
    def test_explicit_tensor_shard_count_matches_requested_gpu_count(self):
        spec = engines.config.BackendSpec(
            backend_id="test_tp", display="test", kind="gpu", ts_backend="ggml_cuda",
            ts_tp=True, ts_env={"TS_DSV41_TP": "{tp}", "TS_DSV4_UBATCH": "256"})
        model = SimpleNamespace(gguf=Path("/tmp/model.gguf"), mmproj=None,
                                is_diffusion=False)
        server = engines.TensorSharpServer(model, "test_tp", Path("/tmp/unused.log"), tp=4)
        with patch.object(engines, "_port_open", return_value=False), \
             patch.object(engines.config, "BACKENDS", {"test_tp": spec}), \
             patch.object(engines.config, "TENSORSHARP_SERVER_DLL", Path("/tmp/server.dll")), \
             patch.object(engines.config, "tp_device_env", return_value={"CUDA_VISIBLE_DEVICES": "1,2,5,7"}), \
             patch.object(server, "_spawn") as spawn:
            server.start()
        arguments, kwargs = spawn.call_args
        command = arguments[0]
        self.assertEqual(command[command.index("--tp") + 1], "4")
        self.assertEqual(kwargs["env"]["TS_DSV41_TP"], "4")
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1,2,5,7")
        self.assertEqual(kwargs["env"]["TS_DSV4_UBATCH"], "256")


class ListenAddressAndPromptParityTests(unittest.TestCase):
    """What every TensorSharp launch carries, whatever the axes."""

    def test_the_configured_port_reaches_the_server(self):
        # The health checks poll 127.0.0.1:<TENSORSHARP_PORT>; before --host/--port
        # were passed, any port but 5000 polled a port nobody listened on.
        cmd = tensorsharp_command()
        self.assertEqual(cmd[cmd.index("--port") + 1], str(engines.config.TENSORSHARP_PORT))
        self.assertEqual(cmd[cmd.index("--host") + 1], "127.0.0.1")

    def test_a_squatted_port_fails_fast_and_says_how_to_move(self):
        model = SimpleNamespace(gguf=Path("/tmp/model.gguf"), mmproj=None, is_diffusion=False)
        spec = engines.config.BackendSpec(
            backend_id="test_moe", display="test", kind="gpu", ts_backend="ggml_cuda")
        server = engines.TensorSharpServer(model, "test_moe", Path("/tmp/unused.log"))
        with patch.object(engines, "_port_open", return_value=True), \
             patch.object(engines, "_pid_listening", return_value=4242), \
             patch.object(engines.config, "BACKENDS", {"test_moe": spec}):
            with self.assertRaises(RuntimeError) as raised:
                server.start()
        self.assertIn("BENCH_TS_PORT", str(raised.exception))

    def test_sub_agent_delegation_is_off_so_prompts_match_the_other_engines(self):
        self.assertIn("--no-multi-agent", tensorsharp_command())


class CpuMoeOffloadTests(unittest.TestCase):
    """`--n-cpu-moe N` / `--cpu-moe-threads M` as a launch axis."""

    def test_baseline_point_launches_the_command_it_always_did(self):
        # The whole point of defaulting the axis off: a run that does not use it
        # must produce a byte-identical server command line.
        never_asked = tensorsharp_command()
        explicit_off = tensorsharp_command(cpu_moe=engines.config.CpuMoeSpec())
        self.assertEqual(never_asked, explicit_off)
        self.assertNotIn("--n-cpu-moe", never_asked)
        self.assertNotIn("--cpu-moe-threads", never_asked)
        self.assertNotIn("--n-cpu-moe", llama_command())

    def test_layer_count_and_host_threads_reach_the_tensorsharp_command_line(self):
        command = tensorsharp_command(
            cpu_moe=engines.config.CpuMoeSpec(layers=8, threads=48))
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "8")
        self.assertEqual(command[command.index("--cpu-moe-threads") + 1], "48")

    def test_all_layers_is_spelled_the_way_the_server_spells_it(self):
        command = tensorsharp_command(
            cpu_moe=engines.config.CpuMoeSpec(layers=engines.config.CPU_MOE_ALL))
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "all")
        # No thread count asked for, so none is sent and the engine's own
        # default (half the usable parallelism) stands.
        self.assertNotIn("--cpu-moe-threads", command)

    def test_llamacpp_gets_the_same_layer_flag_and_its_own_thread_flag(self):
        command = llama_command(
            cpu_moe=engines.config.CpuMoeSpec(layers=4, threads=32))
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "4")
        self.assertEqual(command[command.index("--threads") + 1], "32")

    def test_llamacpp_spells_every_layer_with_its_own_switch(self):
        # `all` is a TensorSharp-only value for --n-cpu-moe: llama.cpp parses
        # that flag's argument as an integer and has a separate switch for every
        # layer, so sending it "all" would abort llama-server at startup.
        command = llama_command(
            cpu_moe=engines.config.CpuMoeSpec(layers=engines.config.CPU_MOE_ALL))
        self.assertIn("--cpu-moe", command)
        self.assertNotIn("--n-cpu-moe", command)
        self.assertNotIn("all", command)

    def test_offload_composes_with_tensor_parallelism(self):
        command = tensorsharp_command(
            tp=4, cpu_moe=engines.config.CpuMoeSpec(layers=engines.config.CPU_MOE_ALL))
        self.assertEqual(command[command.index("--tp") + 1], "4")
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "all")


class CpuMoeAxisTests(unittest.TestCase):
    def test_threads_only_multiply_the_offloaded_points(self):
        axis = engines.config.cpu_moe_axis(["off", "8"], [0, 48])
        self.assertEqual([(c.layers, c.threads) for c in axis],
                         [(0, 0), (8, 0), (8, 48)])

    def test_layer_vocabulary_matches_the_server_flag(self):
        parse = engines.config.parse_cpu_moe_layers
        self.assertEqual(parse("off"), 0)
        self.assertEqual(parse("0"), 0)
        self.assertEqual(parse("all"), engines.config.CPU_MOE_ALL)
        self.assertEqual(parse("16"), 16)
        # One spelling per meaning: `off` is the baseline point's only word, so
        # a near-miss is an error rather than a silently un-offloaded cell.
        for bad in ("some", "none", "-2", "8layers", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse(bad)

    def test_result_filenames_keep_the_baseline_name_and_suffix_the_rest(self):
        import run_matrix
        results = Path("/tmp/results")
        baseline = run_matrix._result_path(results, "tensorsharp", "ggml_cuda",
                                           "m", "text_short")
        self.assertEqual(baseline.name, "tensorsharp__ggml_cuda__m__text_short.json")
        self.assertEqual(
            run_matrix._result_path(results, "tensorsharp", "ggml_cuda", "m",
                                    "text_short", False, 1, 1,
                                    engines.config.CpuMoeSpec()).name,
            baseline.name)
        self.assertEqual(
            run_matrix._result_path(results, "tensorsharp", "ggml_cuda", "m",
                                    "text_short", False, 4, 2,
                                    engines.config.CpuMoeSpec(8, 48)).name,
            "tensorsharp__ggml_cuda__m__text_short__tp2__ncmoe8t48__c4.json")
        self.assertEqual(
            run_matrix._result_path(results, "tensorsharp", "ggml_cuda", "m",
                                    "text_short", False, 1, 1,
                                    engines.config.CpuMoeSpec(engines.config.CPU_MOE_ALL)).name,
            "tensorsharp__ggml_cuda__m__text_short__ncmoeall.json")


class CpuMoeGatingTests(unittest.TestCase):
    """Combinations that cannot offload are recorded as skips with a reason."""

    def setUp(self):
        self.scenario = engines.config.ScenarioSpec(
            short_id="text_short", kind="text", description="")
        self.model = engines.config.ModelSpec(
            short_id="m", display="M", family="f", gguf=Path("/tmp/m.gguf"),
            mmproj=None, modalities={"text"}, size_class="medium")
        self.offload = engines.config.CpuMoeSpec(layers=8)

    def _applies(self, engine, backend, **kw):
        return engines.config.applies(engine, backend, kw.pop("model", self.model),
                                      self.scenario, cpu_moe=self.offload, **kw)

    def test_cpu_backend_cannot_offload_what_is_already_on_the_host(self):
        backends = {"cpu_only": engines.config.BackendSpec(
            backend_id="cpu_only", display="CPU", kind="cpu",
            ts_backend="ggml_cpu", llama_ngl=0)}
        with patch.object(engines.config, "BACKENDS", backends):
            ok, why = self._applies("tensorsharp", "cpu_only")
        self.assertFalse(ok)
        self.assertIn("already host-resident", why)

    def test_connect_only_engine_is_not_driven_into_offload(self):
        backends = {"gpu": engines.config.BackendSpec(
            backend_id="gpu", display="GPU", kind="gpu", ts_backend="ggml_cuda",
            llama_ngl=999, vllm=True)}
        with patch.object(engines.config, "BACKENDS", backends):
            ok, why = self._applies("vllm", "gpu")
        self.assertFalse(ok)
        self.assertIn("MoE CPU offload", why)

    def test_only_an_explicit_dense_declaration_gates_a_model_out(self):
        backends = {"gpu": engines.config.BackendSpec(
            backend_id="gpu", display="GPU", kind="gpu", ts_backend="ggml_cuda")}
        dense = replace(self.model, is_moe=False)
        moe = replace(self.model, is_moe=True)
        with patch.object(engines.config, "BACKENDS", backends):
            self.assertTrue(self._applies("tensorsharp", "gpu")[0])   # unspecified: runs
            self.assertTrue(self._applies("tensorsharp", "gpu", model=moe)[0])
            ok, why = self._applies("tensorsharp", "gpu", model=dense)
        self.assertFalse(ok)
        self.assertIn("is_moe: false", why)

    def test_a_backend_that_already_pins_the_flag_refuses_the_axis(self):
        # benchmark_config_deepseek41.json's `ggml_cuda_layer_cpu_moe4` hardcodes
        # `--n-cpu-moe 4` for both engines. Sweeping the axis on top of it would
        # pass the flag twice and the engine would keep whichever it parsed last,
        # so the cell must be refused by name instead of recorded under the wrong
        # offload point. The baseline point is untouched: that is how this
        # backend has always been measured.
        pinned = {"pinned": engines.config.BackendSpec(
            backend_id="pinned", display="pinned", kind="gpu",
            ts_backend="ggml_cuda", ts_extra_args=("--n-cpu-moe", "4"),
            llama_ngl=999, llama_extra_args=("--n-cpu-moe", "4"))}
        with patch.object(engines.config, "BACKENDS", pinned):
            for engine in ("tensorsharp", "llamacpp"):
                with self.subTest(engine=engine):
                    ok, why = self._applies(engine, "pinned")
                    self.assertFalse(ok)
                    self.assertIn("--n-cpu-moe", why)
                    self.assertTrue(engines.config.applies(
                        engine, "pinned", self.model, self.scenario)[0])

    def test_a_pinned_host_thread_count_only_collides_when_one_is_sent(self):
        # llama.cpp's host-thread knob is its global --threads, which a backend
        # may legitimately pin for reasons that have nothing to do with MoE. It
        # is only a conflict when the axis is about to send one too.
        pinned = {"pinned": engines.config.BackendSpec(
            backend_id="pinned", display="pinned", kind="gpu",
            llama_ngl=999, llama_extra_args=("--threads", "16"))}
        with patch.object(engines.config, "BACKENDS", pinned):
            self.assertTrue(engines.config.applies(
                "llamacpp", "pinned", self.model, self.scenario,
                cpu_moe=engines.config.CpuMoeSpec(layers=8))[0])
            ok, why = engines.config.applies(
                "llamacpp", "pinned", self.model, self.scenario,
                cpu_moe=engines.config.CpuMoeSpec(layers=8, threads=48))
        self.assertFalse(ok)
        self.assertIn("--threads", why)

    def test_the_baseline_point_gates_exactly_as_it_did_before(self):
        backends = {"cpu_only": engines.config.BackendSpec(
            backend_id="cpu_only", display="CPU", kind="cpu", ts_backend="ggml_cpu")}
        dense = replace(self.model, is_moe=False)
        with patch.object(engines.config, "BACKENDS", backends):
            for model in (self.model, dense):
                with self.subTest(is_moe=model.is_moe):
                    self.assertTrue(engines.config.applies(
                        "tensorsharp", "cpu_only", model, self.scenario)[0])
