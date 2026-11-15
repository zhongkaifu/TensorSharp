// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// A recorded step moved to another set of bound buffers (another slot's caches and states) without recording again:
    /// every launch argument that points into one of the recording's bound ranges - a kernel parameter word, a memset's
    /// destination, a device copy's ends - is rewritten in the instantiated graph to the same offset in its counterpart.
    /// The direct engines' graphs serve every slot through pointer tables; a recorded runtime step had bound one slot's
    /// buffers, so every request on a new slot recorded each of its widths again (~30 ms each for Qwen3.8 Flash Next on an
    /// A40). Refused (the caller records instead) wherever an address could be missed: a kernel whose parameters cannot be
    /// read, a child graph, a copy from host memory whose bytes hold a bound address, a node type it does not know.
    /// </summary>
    internal sealed class CudaGraphRelocation
    {
        private readonly struct Range
        {
            public readonly long Base, Bytes;
            public Range(long b, long n) { Base = b; Bytes = n; }
        }

        /// <summary>A kernel parameter word that held a bound address: (byte offset in the parameter buffer, range, delta).</summary>
        private readonly struct Site
        {
            public readonly int Offset, Range;
            public readonly long Delta;
            public Site(int offset, int range, long delta) { Offset = offset; Range = range; Delta = delta; }
        }

        private sealed class KernelNode
        {
            public IntPtr Handle;
            public CudaKernelNodeParams Params;
            public byte[] Buffer;                 // every parameter's recorded bytes, at its offset
            public (int Offset, int Size)[] Layout;
            public Site[] Sites;
        }

        private sealed class MemsetNode
        {
            public IntPtr Handle;
            public CudaMemsetNodeParams Params;
            public Site Site;
        }

        private sealed class MemcpyNode
        {
            public IntPtr Handle;
            public CudaMemcpy3D Params;
            public Site? Src, Dst;
        }

        private readonly Range[] _origin;
        private readonly int[] _byBase;   // origin indices by base, for the search
        private readonly List<KernelNode> _kernels = new();
        private readonly List<MemsetNode> _memsets = new();
        private readonly List<MemcpyNode> _memcpys = new();
        private long[] _applied;   // the bases the instantiation points at now (the origin's at first)

        /// <summary>Why the graph cannot move (null when it can).</summary>
        public string Refusal { get; }

        /// <summary>Index <paramref name="graph"/> (the captured template its instantiation was made from) against the bound
        /// ranges it was recorded over, <paramref name="origin"/> as (device base, bytes) in the binder's order.</summary>
        public CudaGraphRelocation(IntPtr graph, IReadOnlyList<(long Base, long Bytes)> origin)
        {
            _origin = new Range[origin.Count];
            for (int i = 0; i < origin.Count; i++) _origin[i] = new Range(origin[i].Base, origin[i].Bytes);
            _byBase = new int[origin.Count];
            for (int i = 0; i < _byBase.Length; i++) _byBase[i] = i;
            Array.Sort(_byBase, (a, b) => _origin[a].Base.CompareTo(_origin[b].Base));
            _applied = new long[origin.Count];
            for (int i = 0; i < origin.Count; i++) _applied[i] = origin[i].Base;
            Refusal = Index(graph);
        }

        /// <summary>Kernel nodes that bind a moved range (diagnostics, tests).</summary>
        public int PatchedNodes => _kernels.Count + _memsets.Count + _memcpys.Count;

        private string Index(IntPtr graph)
        {
            var count = UIntPtr.Zero;
            if (CudaDriverApi.cuGraphGetNodes(graph, null, ref count) != 0) return "the graph's nodes cannot be listed";
            var nodes = new IntPtr[(int)count];
            if (nodes.Length > 0 && CudaDriverApi.cuGraphGetNodes(graph, nodes, ref count) != 0) return "the graph's nodes cannot be listed";
            var layouts = new Dictionary<IntPtr, (int, int)[]>();
            foreach (IntPtr node in nodes)
            {
                if (CudaDriverApi.cuGraphNodeGetType(node, out int type) != 0) return "a node's type cannot be read";
                switch (type)
                {
                    case 0:
                    {
                        string why = IndexKernel(node, layouts);
                        if (why != null) return why;
                        break;
                    }
                    case 1:
                    {
                        string why = IndexMemcpy(node);
                        if (why != null) return why;
                        break;
                    }
                    case 2:
                    {
                        if (CudaDriverApi.cuGraphMemsetNodeGetParams(node, out CudaMemsetNodeParams p) != 0) return "a memset cannot be read";
                        if (Find(p.Destination.ToInt64(), out Site site, 0)) _memsets.Add(new MemsetNode { Handle = node, Params = p, Site = site });
                        break;
                    }
                    case 3: case 5: case 6: case 7:
                        break;   // host functions, empty nodes and events take no device address
                    default:
                        return $"a node of type {type}";
                }
            }
            return null;
        }

        private unsafe string IndexKernel(IntPtr node, Dictionary<IntPtr, (int, int)[]> layouts)
        {
            if (CudaDriverApi.cuGraphKernelNodeGetParams(node, out CudaKernelNodeParams p) != 0) return "a kernel's parameters cannot be read";
            if (p.KernelParams == IntPtr.Zero) return p.Extra == IntPtr.Zero ? null : "a kernel launched with a packed parameter buffer";
            IntPtr handle = p.Function != IntPtr.Zero ? p.Function : p.Kernel;
            if (handle == IntPtr.Zero) return "a kernel node names no kernel";
            if (!layouts.TryGetValue(handle, out (int Offset, int Size)[] layout))
            {
                var list = new List<(int, int)>();
                for (int i = 0; ; i++)
                {
                    int rc = p.Function != IntPtr.Zero
                        ? CudaDriverApi.cuFuncGetParamInfo(p.Function, (UIntPtr)i, out UIntPtr offset, out UIntPtr size)
                        : CudaDriverApi.cuKernelGetParamInfo(p.Kernel, (UIntPtr)i, out offset, out size);
                    // Past the last parameter: CUDA_ERROR_INVALID_VALUE; anything else and the layout is unknown.
                    if (rc == InvalidValue) break;
                    if (rc != 0) return $"a kernel's parameter layout cannot be read (CUDA error {rc})";
                    list.Add(((int)offset, (int)size));
                }
                layout = list.ToArray();
                layouts[handle] = layout;
            }
            int total = 0;
            foreach ((int offset, int size) in layout) total = Math.Max(total, offset + size);
            var buffer = new byte[total];
            var sites = new List<Site>();
            var args = (IntPtr*)p.KernelParams;
            for (int i = 0; i < layout.Length; i++)
            {
                (int offset, int size) = layout[i];
                if (args[i] == IntPtr.Zero) return "a kernel parameter has no value";
                new ReadOnlySpan<byte>((void*)args[i], size).CopyTo(buffer.AsSpan(offset, size));
                // A pointer is 8-byte aligned in its parameter, wherever a structure puts it.
                for (int w = 0; w + 8 <= size; w += 8)
                    if (Find(BitConverter.ToInt64(buffer, offset + w), out Site site, offset + w)) sites.Add(site);
            }
            if (sites.Count > 0)
                _kernels.Add(new KernelNode { Handle = node, Params = p, Buffer = buffer, Layout = layout, Sites = sites.ToArray() });
            return null;
        }

        private unsafe string IndexMemcpy(IntPtr node)
        {
            if (CudaDriverApi.cuGraphMemcpyNodeGetParams(node, out CudaMemcpy3D p) != 0) return "a copy cannot be read";
            if (p.SrcMemoryType is 3 || p.DstMemoryType is 3) return "a copy to or from an array";
            Site? src = null, dst = null;
            if (p.SrcMemoryType is 2 or 4 && Find(p.SrcDevice.ToInt64(), out Site s, 0)) src = s;
            if (p.DstMemoryType is 2 or 4 && Find(p.DstDevice.ToInt64(), out Site d, 0)) dst = d;
            // Bytes copied from the host: a bound address among them would land on the device unmoved.
            if (p.SrcMemoryType == 1 && p.SrcHost != IntPtr.Zero)
            {
                long bytes = (long)p.WidthInBytes * Math.Max(1L, (long)p.Height) * Math.Max(1L, (long)p.Depth);
                var span = new ReadOnlySpan<byte>((void*)(p.SrcHost + (nint)(long)p.SrcXInBytes), (int)Math.Min(bytes, int.MaxValue));
                for (int w = 0; w + 8 <= span.Length; w += 8)
                    if (Find(MemoryMarshal.Read<long>(span.Slice(w)), out _, 0)) return "a host copy carries a bound address";
            }
            if ((src != null || dst != null) && ((long)p.Height > 1 || (long)p.Depth > 1)) return "a pitched copy of a bound range";
            if (src != null || dst != null) _memcpys.Add(new MemcpyNode { Handle = node, Params = p, Src = src, Dst = dst });
            return null;
        }

        private const int InvalidValue = 1;


        /// <summary>The bound range <paramref name="value"/> points into. Its one-past-end is not: the pool carves blocks back
        /// to back, and the next block's pointers (a recording's own) had moved with it after a long test run.</summary>
        private bool Find(long value, out Site site, int offset)
        {
            site = default;
            // The last range whose base is at or below the value.
            int lo = 0, hi = _byBase.Length - 1, at = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >> 1;
                if (_origin[_byBase[mid]].Base <= value) { at = mid; lo = mid + 1; }
                else hi = mid - 1;
            }
            if (at < 0) return false;
            int r = _byBase[at];
            long b = _origin[r].Base, n = _origin[r].Bytes;
            if (value >= b + n) return false;
            site = new Site(offset, r, value - b);
            return true;
        }

        /// <summary>Point the instantiation at <paramref name="bases"/> (one per origin range, in its order; ranges that shared
        /// a base must share one). False when the driver refused a node: the graph is then partly moved and must not run.</summary>
        public unsafe bool Apply(IntPtr exec, IntPtr context, IReadOnlyList<long> bases)
        {
            if (bases.Count != _origin.Length) return false;
            bool same = true;
            for (int i = 0; i < bases.Count && same; i++) same = bases[i] == _applied[i];
            if (same) return true;
            foreach (KernelNode k in _kernels)
            {
                byte[] buffer = (byte[])k.Buffer.Clone();
                foreach (Site s in k.Sites)
                    BitConverter.TryWriteBytes(buffer.AsSpan(s.Offset, 8), bases[s.Range] + s.Delta);
                fixed (byte* bytes = buffer)
                {
                    var args = stackalloc IntPtr[Math.Max(1, k.Layout.Length)];
                    for (int i = 0; i < k.Layout.Length; i++) args[i] = (IntPtr)(bytes + k.Layout[i].Offset);
                    CudaKernelNodeParams p = k.Params;
                    p.KernelParams = (IntPtr)args;
                    p.Extra = IntPtr.Zero;
                    if (CudaDriverApi.cuGraphExecKernelNodeSetParams(exec, k.Handle, p) != 0) return false;
                }
            }
            foreach (MemsetNode m in _memsets)
            {
                CudaMemsetNodeParams p = m.Params;
                p.Destination = (IntPtr)(bases[m.Site.Range] + m.Site.Delta);
                if (CudaDriverApi.cuGraphExecMemsetNodeSetParams(exec, m.Handle, p, context) != 0) return false;
            }
            foreach (MemcpyNode c in _memcpys)
            {
                CudaMemcpy3D p = c.Params;
                if (c.Src is { } s) p.SrcDevice = (IntPtr)(bases[s.Range] + s.Delta);
                if (c.Dst is { } d) p.DstDevice = (IntPtr)(bases[d.Range] + d.Delta);
                if (CudaDriverApi.cuGraphExecMemcpyNodeSetParams(exec, c.Handle, p, context) != 0) return false;
            }
            for (int i = 0; i < bases.Count; i++) _applied[i] = bases[i];
            return true;
        }
    }
}
