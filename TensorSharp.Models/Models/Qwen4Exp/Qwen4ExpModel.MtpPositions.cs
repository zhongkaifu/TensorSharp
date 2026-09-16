// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        // Draft catch-up may run after the target has processed a later prompt
        // chunk. Preserve each chunk's positions/gap under its holder identity;
        // consulting only the latest target gap is then too late.
        internal sealed class MtpPositionRange
        {
            internal int Position, Count, RopePosition;
            internal int[] MultiAxis;
            internal int End => checked(Position + Count);
        }

        private readonly Dictionary<object, List<MtpPositionRange>> _mtpPositions = new(ReferenceEqualityComparer.Instance);

        private MtpPositionRange PrepareMtpPositions(object owner, int count)
        {
            if (!_mtpPositions.TryGetValue(owner, out var ranges))
            {
                ranges = new List<MtpPositionRange>();
                _mtpPositions.Add(owner, ranges);
            }
            ranges.EnsureCapacity(checked(ranges.Count + 1));
            var range = new MtpPositionRange
            {
                Position = _cacheSeqLen,
                Count = count,
                RopePosition = checked((int)Math.Max(0L, (long)_cacheSeqLen - _mropeCacheGap)),
            };
            if (_pendingMRoPEPositions != null)
            {
                int length = checked(3 * count);
                if (_pendingMRoPEPositions.Length < length)
                    throw new ArgumentException("qwen4exp MTP positions do not cover the target chunk.");
                range.MultiAxis = new int[length];
                Array.Copy(_pendingMRoPEPositions, range.MultiAxis, length);
            }
            return range;
        }

        internal static void PublishMtpPositions(List<MtpPositionRange> ranges, MtpPositionRange range)
        {
            // A replay replaces the abandoned tail. Preserve only the older
            // prefix, without copying any position arrays.
            while (ranges.Count != 0 && ranges[^1].Position >= range.Position)
                ranges.RemoveAt(ranges.Count - 1);
            if (ranges.Count != 0 && ranges[^1].End > range.Position)
                ranges[^1].Count = range.Position - ranges[^1].Position;
            ranges.Add(range); // PrepareMtpPositions reserved publication capacity.
        }

        internal static (int RopePosition, int[] MultiAxis) ResolveMtpPositions(
            IReadOnlyList<MtpPositionRange> ranges, int position, int count, int fallbackGap)
        {
            if (position < 0 || count <= 0 || position > int.MaxValue - count)
                throw new ArgumentOutOfRangeException(nameof(position));
            int fallback = checked((int)Math.Max(0L, (long)position - fallbackGap));
            if (ranges == null || ranges.Count == 0) return (fallback, null);
            var latest = ranges[^1];
            if (latest.Position <= position && latest.End >= position + count && latest.MultiAxis == null)
                return (checked(latest.RopePosition + position - latest.Position), null);

            var positions = new int[checked(3 * count)];
            for (int token = 0; token < count; ++token)
            {
                int absolute = position + token;
                MtpPositionRange found = null;
                // Newer ranges win even for a caller-supplied overlapping list.
                for (int i = ranges.Count - 1; i >= 0; --i)
                    if (ranges[i].Position <= absolute && absolute < ranges[i].End)
                    { found = ranges[i]; break; }
                for (int axis = 0; axis < 3; ++axis)
                    positions[3 * token + axis] = found == null ? checked(fallback + token)
                        : found.MultiAxis == null ? checked(found.RopePosition + absolute - found.Position)
                        : found.MultiAxis[3 * (absolute - found.Position) + axis];
            }
            int first = positions[0];
            bool scalar = true;
            for (int token = 0; token < count && scalar; ++token)
                for (int axis = 0; axis < 3; ++axis)
                    scalar &= positions[3 * token + axis] == (long)first + token;
            return scalar ? (first, null) : (first, positions);
        }

        private void PruneMtpPositions(object owner, int consumedPosition)
        {
            if (!_mtpPositions.TryGetValue(owner, out var ranges)) return;
            int consumed = 0;
            while (consumed < ranges.Count && ranges[consumed].End <= consumedPosition) ++consumed;
            if (consumed != 0) ranges.RemoveRange(0, consumed);
        }
    }
}
