// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
namespace TensorSharp.Chat.Studio;
/// <summary>Limits even chunked request bodies without first buffering attacker-controlled input.</summary>
public sealed class LimitedReadStream(Stream source, long maximum) : Stream
{
    private long _read;
    private int Count(int count) { _read += count; if (_read > maximum) throw new ArgumentException("Studio request is too large."); return count; }
    public override int Read(byte[] buffer, int offset, int count) => Count(source.Read(buffer, offset, Math.Min(count, (int)Math.Min(int.MaxValue, maximum - _read + 1))));
    public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken ct = default) => Count(await source.ReadAsync(buffer[..Math.Min(buffer.Length, (int)Math.Min(int.MaxValue, maximum - _read + 1))], ct));
    public override bool CanRead => true; public override bool CanSeek => false; public override bool CanWrite => false;
    public override long Length => throw new NotSupportedException(); public override long Position { get => _read; set => throw new NotSupportedException(); }
    public override void Flush() { } public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
    public override void SetLength(long value) => throw new NotSupportedException(); public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
}
