// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Host metadata only. No CUDA queries, synchronization, token/logit reads or hashes.
using System;
using System.Runtime.CompilerServices;
using System.IO;
using System.Text.Json;
using System.Text;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private readonly ServingTraceLog _servingTrace;
        public bool ServingTraceEnabled => _servingTrace != null;
        public long ServingTraceDroppedEvents => _servingTrace?.DroppedEvents ?? 0;

        /// <summary>Diagnostic ownership supplied by the model under its existing sequence lock.
        /// Null explicitly marks the primary slot as unmapped; it is never guessed from timing.</summary>
        public void TraceBindSequence(string requestId, Slot slot) => _servingTrace?.Bind(requestId, slot);

        private void RunGroups(int tapFirst, params RowGroup[] groups)
        {
            if (_servingTrace == null)
            {
                RunGroupsCore(tapFirst, groups);
                return;
            }
            var capture = _servingTrace.BeginForward(tapFirst, groups, VerifyGraphReplayCount, _verifyDecodeQ81);
            try
            {
                RunGroupsCore(tapFirst, groups);
                _servingTrace.EndForward(capture, groups, VerifyGraphReplayCount, null);
            }
            catch (Exception error)
            {
                _servingTrace.EndForward(capture, groups, VerifyGraphReplayCount, error.GetType().Name);
                throw;
            }
        }

        // This ledger exists only after an exact "1" opt-in. The disabled hot path is a null
        // guard: no dictionary, writer, snapshots, strings, counters or arrays are created.
        internal sealed class ServingTraceLog : IDisposable
        {
            internal const string Schema = "tensorsharp-q35-serving-trace/1";
            internal const string FileSchema = "tensorsharp-q35-serving-trace/2";
            private readonly TextWriter _writer;
            private readonly object _writeGate = new();
            private readonly bool _ownsWriter;
            private bool _closed;
            private readonly string _engineId = Guid.NewGuid().ToString("N");
            private readonly ConditionalWeakTable<Slot, Binding> _bindings = new();
            private long _nextSlot, _nextForward, _event, _lastForward;
            private ForwardCapture _current;
            internal long DroppedEvents { get; private set; }

            internal static ServingTraceLog FromEnvironment()
            {
                if (Environment.GetEnvironmentVariable("TS_Q35_SERVING_TRACE") != "1") return null;
                string file = Environment.GetEnvironmentVariable("TS_Q35_SERVING_TRACE_FILE");
                return file == null ? Create("1", Console.Error) : CreateFile(file);
            }
            internal static ServingTraceLog Create(string option, TextWriter writer) =>
                option == "1" ? new ServingTraceLog(writer) : null;
            private ServingTraceLog(TextWriter writer, bool ownsWriter = false)
                => (_writer, _ownsWriter) = (writer, ownsWriter);

            internal static ServingTraceLog CreateFile(string path)
            {
                ValidateFilePath(path);
                FileStream stream = null;
                StreamWriter writer = null;
                try
                {
                    stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                    writer = new StreamWriter(stream, new UTF8Encoding(false)) { AutoFlush = true };
                    var log = new ServingTraceLog(writer, ownsWriter: true);
                    log.Emit("trace_opened", new { sink = "dedicated_file", file_path = path, terminal_marker_required = true });
                    if (log.DroppedEvents != 0) throw new IOException("Serving trace opening marker could not be written.");
                    return log;
                }
                catch
                {
                    try { writer?.Dispose(); } catch { }
                    try { stream?.Dispose(); } catch { }
                    throw; // Explicit file configuration never falls back to stdout/stderr.
                }
            }
            private static void ValidateFilePath(string path)
            {
                if (string.IsNullOrWhiteSpace(path) || !Path.IsPathFullyQualified(path) || Path.GetFullPath(path) != path)
                    throw new ArgumentException("Serving trace file must be an absolute normalized path.", nameof(path));
                foreach (char c in path)
                    if (char.IsControl(c)) throw new ArgumentException("Control characters in serving trace file path.", nameof(path));
                if (File.Exists(path) || Directory.Exists(path) || IsLink(path)) throw new IOException("Serving trace file must be fresh.");
                string parent = Path.GetDirectoryName(path);
                if (!Directory.Exists(parent)) throw new DirectoryNotFoundException("Serving trace parent must already exist.");
                for (string p = parent; p != null; p = Path.GetDirectoryName(p))
                    if (IsLink(p)) throw new IOException("Serving trace file path must not contain aliases.");
            }
            private static bool IsLink(string path) => new FileInfo(path).LinkTarget != null || new DirectoryInfo(path).LinkTarget != null;

            public void Dispose() => Close("trace_disposed");
            internal void Close(string reason)
            {
                lock (_writeGate)
                {
                    if (_closed) return;
                    if (_ownsWriter)
                    {
                        try { _writer.Flush(); } catch (Exception) { DroppedEvents++; }
                        EmitCore("trace_closed", new
                        {
                            reason, last_event_index = _event, last_forward_id = _nextForward,
                            last_successful_forward_id = _lastForward, active_forward_id = _current?.Id,
                            dropped_events = DroppedEvents
                        });
                        _closed = true;
                        try { _writer.Dispose(); } catch (Exception) { DroppedEvents++; }
                    }
                    else _closed = true; // Never dispose the inherited Console.Error writer.
                }
            }

            private sealed class Binding
            {
                internal long Id, Generation, LastVerify;
                internal string Request;
                internal bool Bound;
            }
            private Binding Identity(Slot slot)
            {
                if (!_bindings.TryGetValue(slot, out var value))
                {
                    value = new Binding { Id = ++_nextSlot };
                    _bindings.Add(slot, value);
                }
                return value;
            }
            private static string BoundedId(string value) => value == null || value.Length <= 128 ? value : value[..128];
            internal void Bind(string request, Slot slot)
            {
                if (slot == null) return;
                var value = Identity(slot);
                if (value.Bound && string.Equals(value.Request, request, StringComparison.Ordinal)) return;
                value.Bound = true;
                value.Request = request;
                value.Generation++;
                value.LastVerify = 0;
                Emit("sequence_slot_binding", new
                {
                    slot_id = value.Id, binding_generation = value.Generation,
                    engine_sequence_id = BoundedId(request), request_id_truncated = request?.Length > 128,
                    binding_status = request == null ? "primary_unmapped" : "mapped",
                    npast = slot.NPast, rope_delta = slot.RopeDelta, block_table_version = slot.Version
                });
            }
            internal sealed class ForwardCapture
            {
                internal long Id, GraphReplaysBefore;
                internal int TapFirst, TotalRows;
                internal bool Q81PolicyEnabled;
                internal object[] GroupsBefore;
                internal string LayerBranch;
            }
            private object SlotMetadata(Slot slot, int realRows)
            {
                var value = Identity(slot);
                return new
                {
                    slot_id = value.Id, binding_generation = value.Generation,
                    engine_sequence_id = BoundedId(value.Request), request_id_truncated = value.Request?.Length > 128,
                    binding_status = value.Request == null ? "unmapped" : "mapped", real_rows = realRows,
                    npast = slot.NPast, rope_delta = slot.RopeDelta, block_table_version = slot.Version
                };
            }
            private object[] GroupsMetadata(RowGroup[] groups)
            {
                var result = new object[groups.Length];
                for (int i = 0; i < groups.Length; i++)
                {
                    var group = groups[i];
                    var slots = new object[group.Slots.Length];
                    int realRows = 0;
                    for (int q = 0; q < slots.Length; q++)
                    {
                        int rows = group.Counts == null ? group.RowsPerSeq : group.Counts[q];
                        realRows += rows;
                        slots[q] = SlotMetadata(group.Slots[q], rows);
                    }
                    result[i] = new
                    {
                        group_ordinal = i, kind = group.Verify ? "verify" : group.Decode ? "decode" : "prefill",
                        sequences = group.Slots.Length, rows_per_sequence = group.RowsPerSeq,
                        padded_rows = group.Rows, real_rows = realRows,
                        ragged_counts_present = group.Counts != null, slots
                    };
                }
                return result;
            }
            internal ForwardCapture BeginForward(int tapFirst, RowGroup[] groups, long graphReplays, bool q81PolicyEnabled = false)
            {
                var capture = new ForwardCapture
                {
                    Id = ++_nextForward, TapFirst = tapFirst, GraphReplaysBefore = graphReplays, Q81PolicyEnabled = q81PolicyEnabled,
                    GroupsBefore = GroupsMetadata(groups)
                };
                foreach (var group in groups) capture.TotalRows += group.Rows;
                _current = capture;
                return capture;
            }
            // Called exactly at the actual host layer-loop branch, outside the layer loop.
            // A graph replay that does not re-enter it remains explicitly unobserved.
            internal void LayerBranch(string branch)
            {
                if (_current != null) _current.LayerBranch = branch;
            }
            internal void EndForward(ForwardCapture capture, RowGroup[] groups, long graphReplays, string failure)
            {
                _current = null;
                _lastForward = failure == null ? capture.Id : 0;
                if (failure == null)
                    foreach (var group in groups)
                        if (group.Verify)
                            foreach (var slot in group.Slots) Identity(slot).LastVerify = capture.Id;
                Emit("run_groups", new
                {
                    forward_id = capture.Id, total_rows = capture.TotalRows, tap_first = capture.TapFirst,
                    groups_before = capture.GroupsBefore, groups_after = GroupsMetadata(groups),
                    verify_decode_Q81_enabled = capture.Q81PolicyEnabled,
                    layer_branch_observed = capture.LayerBranch != null,
                    layer_branch = capture.LayerBranch ?? "NOT_OBSERVED_ON_HOST",
                    verify_graph_replay_delta = graphReplays - capture.GraphReplaysBefore,
                    outcome = failure == null ? "host_submission_returned" : "failed", failure_type = failure,
                    device_completion = "NOT_QUERIED_BY_TRACE"
                });
            }
            internal void CommitCompleted(Slot[] slots, int window, int[] kept, int[] before)
            {
                var rows = new object[slots.Length];
                for (int i = 0; i < rows.Length; i++)
                {
                    var id = Identity(slots[i]);
                    rows[i] = new { slot_id = id.Id, binding_generation = id.Generation,
                        engine_sequence_id = BoundedId(id.Request), verify_forward_id = id.LastVerify,
                        kept = kept[i], npast_before = before[i], npast_after = slots[i].NPast };
                }
                Emit("verify_commit_completed", new { window, slots = rows, completion = "existing_sync_returned" });
            }
            internal void HeadCompleted(int rows, int vocab, bool logitsCopied, bool argmaxCopied)
            {
                Emit("head_host_return_completed", new
                {
                    forward_id = _lastForward, rows, vocab,
                    logits_bytes_copied = logitsCopied ? (long)rows * vocab * sizeof(float) : 0,
                    argmax_bytes_copied = argmaxCopied ? (long)rows * sizeof(int) : 0,
                    completion = "existing_sync_and_requested_host_copies_returned"
                });
            }
            private void Emit(string kind, object metadata)
            {
                lock (_writeGate)
                {
                    if (_closed) { DroppedEvents++; return; }
                    EmitCore(kind, metadata);
                }
            }
            private void EmitCore(string kind, object metadata)
            {
                try
                {
                    string json = JsonSerializer.Serialize(new
                    {
                        schema = _ownsWriter ? FileSchema : Schema, kind, engine_id = _engineId, event_index = ++_event,
                        utc = DateTimeOffset.UtcNow, previously_dropped_events = DroppedEvents, metadata
                    });
                    _writer.WriteLine(_ownsWriter ? json : "[q35-serving-trace] " + json);
                }
                catch (Exception) { DroppedEvents++; } // Diagnostic I/O must not fail a model operation.
            }
        }
    }
}
